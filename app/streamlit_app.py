import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
import streamlit as st

from src.api.inference_service import PredictionResult, SegmentationService
from src.api.settings import Settings

DEFAULT_API_URL = os.getenv("UNET3D_API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 300


@dataclass
class InferenceBundle:
    volume: np.ndarray
    mask: np.ndarray
    mask_bytes: bytes
    summary: Dict[str, Any]


st.set_page_config(page_title="UNet3D Portfolio Demo", page_icon="CT", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #f4efe6 0%, #dcefe8 100%);
        border: 1px solid #cbded5;
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        color: #143642;
        margin-bottom: 1rem;
    }
    .hero-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2f6f62;
        margin-bottom: 0.25rem;
    }
    .hero-title {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-copy {
        font-size: 1rem;
        color: #35515a;
        max-width: 60rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-kicker">Medical Imaging Portfolio</div>
      <div class="hero-title">UNet3D for volumetric segmentation, packaged like a product.</div>
      <div class="hero-copy">
        Evalua un volumen NIfTI contra la API o con inferencia local, inspecciona metadata del modelo,
        revisa overlays por corte y descarga la mascara final sin salir del panel.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _tmp_suffix(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"


def _load_nifti_from_bytes(raw: bytes, suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        image = nib.load(str(tmp_path))
        canonical = nib.as_closest_canonical(image)
        data = canonical.get_fdata().astype(np.float32)
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        return data, canonical.affine
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def _render_overlay(volume: np.ndarray, mask: np.ndarray, axis: int, index: int, key_prefix: str) -> None:
    image_slice = _extract_slice(volume, axis, index)
    mask_slice = _extract_slice(mask, axis, index)
    overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Input slice")
    axes[1].imshow(mask_slice, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Predicted mask")
    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(overlay, cmap="autumn", alpha=0.45, interpolation="nearest")
    axes[2].set_title("Overlay")

    for axis_obj in axes:
        axis_obj.axis("off")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _summary_from_local_result(result: PredictionResult, settings: Settings) -> Dict[str, Any]:
    return {
        "request_id": "local-execution",
        "input_filename": result.input_filename,
        "output_filename": result.output_filename,
        "input_shape": result.input_shape,
        "padded_shape": result.padded_shape,
        "voxel_spacing": result.voxel_spacing,
        "orientation": result.orientation,
        "device": result.device,
        "threshold_used": result.threshold_used,
        "runtime": {
            "total_ms": result.total_runtime_ms,
            "preprocess_ms": result.preprocess_ms,
            "inference_ms": result.inference_ms,
            "postprocess_ms": result.postprocess_ms,
        },
        "preprocessing": {
            "clip_percentiles": settings.clip_percentiles,
            "pad_multiple": settings.pad_multiple,
            "input_intensity_range": result.intensity_range,
        },
        "stats": {
            "voxel_count": result.voxel_count,
            "labels_present": result.labels_present,
            "class_histogram": result.class_histogram,
            "class_ratios": result.class_ratios,
        },
    }


def _histogram_rows(summary: Dict[str, Any]) -> list[Dict[str, Any]]:
    histogram = summary["stats"]["class_histogram"]
    ratios = summary["stats"]["class_ratios"]
    rows = []
    for label, count in histogram.items():
        rows.append(
            {
                "class_id": int(label),
                "voxels": int(count),
                "ratio": float(ratios.get(label, ratios.get(str(label), 0.0))),
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def _get_local_service(model_path: str) -> SegmentationService:
    settings = Settings(model_path=Path(model_path))
    service = SegmentationService(settings=settings)
    service.load_model()
    return service


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_api_status(api_url: str) -> Dict[str, Any]:
    try:
        health_response = requests.get(f"{api_url.rstrip('/')}/health/ready", timeout=15)
        health_response.raise_for_status()
        model_response = requests.get(f"{api_url.rstrip('/')}/api/v1/model", timeout=15)
        model_response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": str(exc)}

    return {"health": health_response.json(), "model": model_response.json()}


def _run_api_inference(api_url: str, uploaded_file, threshold: Optional[float]) -> InferenceBundle:
    file_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
    params = {}
    if threshold is not None:
        params["threshold"] = threshold

    with st.spinner("Consultando la API y descargando la mascara..."):
        summary_response = requests.post(
            f"{api_url.rstrip('/')}/api/v1/predictions",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS,
        )
        summary_response.raise_for_status()

        download_response = requests.post(
            f"{api_url.rstrip('/')}/api/v1/predictions/download",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS,
        )
        download_response.raise_for_status()

    volume, _ = _load_nifti_from_bytes(file_bytes, _tmp_suffix(uploaded_file.name))
    mask, _ = _load_nifti_from_bytes(download_response.content, ".nii.gz")
    summary = summary_response.json()
    return InferenceBundle(volume=volume, mask=mask.astype(np.uint8), mask_bytes=download_response.content, summary=summary)


def _run_local_inference(model_path: str, uploaded_file, threshold: Optional[float]) -> InferenceBundle:
    service = _get_local_service(model_path)
    file_bytes = uploaded_file.getvalue()
    suffix = _tmp_suffix(uploaded_file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Ejecutando inferencia local con UNet3D..."):
            result = service.predict(tmp_path, threshold=threshold)
    finally:
        tmp_path.unlink(missing_ok=True)

    volume, _ = _load_nifti_from_bytes(file_bytes, suffix)
    summary = _summary_from_local_result(result, service.settings)
    return InferenceBundle(volume=volume, mask=result.mask, mask_bytes=result.mask_bytes, summary=summary)


def _render_result(bundle: InferenceBundle, key_prefix: str) -> None:
    summary = bundle.summary
    stats = summary["stats"]
    runtime = summary["runtime"]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Volume shape", "x".join(str(dim) for dim in summary["input_shape"]))
    metric_2.metric("Labels found", len(stats["labels_present"]))
    metric_3.metric("Runtime", f"{runtime['total_ms']:.1f} ms")
    metric_4.metric("Device", summary["device"])

    controls_left, controls_right = st.columns([0.7, 1.3])
    with controls_left:
        axis = st.selectbox("Axis", options=[0, 1, 2], index=2, key=f"{key_prefix}_axis")
        axis_length = bundle.mask.shape[axis]
        index = st.slider(
            "Slice index",
            min_value=0,
            max_value=max(0, axis_length - 1),
            value=axis_length // 2,
            key=f"{key_prefix}_slice",
        )
        st.download_button(
            "Download mask",
            data=bundle.mask_bytes,
            file_name=summary["output_filename"],
            mime="application/gzip",
            key=f"{key_prefix}_download",
        )
        st.caption(f"Request ID: `{summary['request_id']}`")

    with controls_right:
        _render_overlay(bundle.volume, bundle.mask, axis=axis, index=index, key_prefix=key_prefix)

    details_col, histogram_col = st.columns([1.2, 1.0])
    with details_col:
        with st.expander("Prediction metadata", expanded=False):
            st.json(summary)
    with histogram_col:
        st.subheader("Class histogram")
        st.table(_histogram_rows(summary))


def _handle_api_tab() -> None:
    st.subheader("Remote inference via FastAPI")
    status = _fetch_api_status(st.session_state["api_url"])

    if "error" in status:
        st.warning(f"No se pudo consultar la API: {status['error']}")
    else:
        health = status["health"]
        model = status["model"]
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric("API status", health["status"])
        col_2.metric("Ready", "yes" if health["ready"] else "no")
        col_3.metric("Device", health["device"])
        col_4.metric("Classes", model["num_classes"])
        with st.expander("Model metadata", expanded=False):
            st.json(model)

    uploaded_file = st.file_uploader("NIfTI volume (.nii, .nii.gz)", type=["nii", "nii.gz"], key="api_uploader")
    if st.button("Run API inference", type="primary", key="api_infer_button"):
        if not uploaded_file:
            st.warning("Sube un archivo NIfTI para comenzar.")
        else:
            try:
                st.session_state["api_bundle"] = _run_api_inference(
                    api_url=st.session_state["api_url"],
                    uploaded_file=uploaded_file,
                    threshold=st.session_state["threshold"],
                )
            except requests.RequestException as exc:
                st.error(f"La API devolvio un error: {exc}")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"No se pudo completar la inferencia remota: {exc}")

    bundle = st.session_state.get("api_bundle")
    if bundle is not None:
        _render_result(bundle, key_prefix="api_result")


def _handle_local_tab(default_settings: Settings) -> None:
    st.subheader("Local inference on the same machine")
    model_path = st.text_input("Checkpoint path", value=st.session_state["local_model_path"])
    st.session_state["local_model_path"] = model_path

    helper_left, helper_right = st.columns([0.8, 1.2])
    with helper_left:
        if st.button("Clear local model cache", key="clear_local_cache"):
            _get_local_service.clear()
            st.info("Cache de modelo local limpiada.")
    with helper_right:
        st.caption(
            "Usa este modo para validar el checkpoint sin depender de la API. "
            "La app reutiliza el modelo en cache para que las corridas siguientes sean mas rapidas."
        )

    uploaded_file = st.file_uploader("NIfTI volume (.nii, .nii.gz)", type=["nii", "nii.gz"], key="local_uploader")
    if st.button("Run local inference", type="primary", key="local_infer_button"):
        if not uploaded_file:
            st.warning("Sube un archivo NIfTI para comenzar.")
        else:
            try:
                st.session_state["local_bundle"] = _run_local_inference(
                    model_path=model_path,
                    uploaded_file=uploaded_file,
                    threshold=st.session_state["threshold"],
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"No se pudo completar la inferencia local: {exc}")

    bundle = st.session_state.get("local_bundle")
    if bundle is not None:
        _render_result(bundle, key_prefix="local_result")
    elif Path(default_settings.model_path).exists():
        st.info("El checkpoint por defecto existe. Puedes correr inferencia local apenas subas un volumen.")
    else:
        st.info("Configura una ruta valida a tu checkpoint para habilitar la inferencia local.")


default_settings = Settings()
st.session_state.setdefault("api_url", DEFAULT_API_URL)
st.session_state.setdefault("threshold", float(default_settings.default_threshold))
st.session_state.setdefault("local_model_path", str(default_settings.model_path))

sidebar = st.sidebar
sidebar.header("Execution Controls")
st.session_state["api_url"] = sidebar.text_input("API base URL", value=st.session_state["api_url"])
st.session_state["threshold"] = sidebar.slider(
    "Binary threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["threshold"]),
    step=0.05,
)
sidebar.caption(
    "La API usa el endpoint JSON para metadata y el endpoint de descarga para traer la mascara NIfTI."
)

tab_api, tab_local = st.tabs(["API Deployment", "Local Checkpoint"])

with tab_api:
    _handle_api_tab()

with tab_local:
    _handle_local_tab(default_settings)
