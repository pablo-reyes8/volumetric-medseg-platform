import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np


def basename_noext(filename: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", filename)


def load_nifti_volume(
    nifti_path: Path | str,
    dtype=np.float32,
    ensure_3d: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    image = nib.load(str(nifti_path))
    canonical = nib.as_closest_canonical(image)
    data = canonical.get_fdata().astype(dtype)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if ensure_3d and data.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D; shape recibido {data.shape}")
    spacing = tuple(float(value) for value in canonical.header.get_zooms()[: data.ndim])
    return data, canonical.affine, spacing


def minmax_normalize(img: np.ndarray, clip=None, eps: float = 1e-6) -> np.ndarray:
    x = img.astype(np.float32)
    if clip is not None:
        low, high = np.percentile(x, clip)
        if high - low < eps:
            low, high = x.min(), x.max()
        x = np.clip(x, low, high)

    min_value = float(x.min())
    max_value = float(x.max())
    if max_value - min_value < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_value) / (max_value - min_value)


def quick_meta(nifti_path: Path | str) -> Tuple[Tuple[int, ...], Tuple[float, ...], np.dtype]:
    image = nib.load(str(nifti_path))
    return image.shape, image.header.get_zooms(), image.get_data_dtype()


def quick_subsample_stats(nifti_path: Path | str, step: int = 4) -> Dict[str, float | bool]:
    image = nib.load(str(nifti_path))
    array = np.asanyarray(image.dataobj)[::step, ::step, ::step]
    if array.size == 0:
        return {"empty": True, "finite_ok": False, "var": 0.0}
    finite = np.isfinite(array)
    variance = float(array.var()) if finite.any() else 0.0
    return {"empty": False, "finite_ok": bool(finite.all()), "var": variance}


def summarize_volume(nifti_path: Path | str, step: int = 4) -> Dict[str, object]:
    volume, _, spacing = load_nifti_volume(nifti_path)
    subsample = volume[::step, ::step, ::step]
    finite = np.isfinite(subsample)
    finite_values = subsample[finite]
    return {
        "path": str(nifti_path),
        "shape": tuple(int(dim) for dim in volume.shape),
        "spacing": tuple(float(value) for value in spacing[:3]),
        "dtype": str(volume.dtype),
        "min": float(finite_values.min()) if finite_values.size else 0.0,
        "max": float(finite_values.max()) if finite_values.size else 0.0,
        "mean": float(finite_values.mean()) if finite_values.size else 0.0,
        "std": float(finite_values.std()) if finite_values.size else 0.0,
        "finite_ok": bool(finite.all()),
    }


def check_files(dir_path: Path | str, files: Iterable[str], label_mode: bool = False, max_list: int = 5):
    directory = Path(dir_path)
    bad = {"size0": [], "corrupt": [], "naninf": [], "zerovar": [], "emptyslice": []}
    for filename in files:
        file_path = directory / filename
        if file_path.stat().st_size == 0:
            bad["size0"].append(filename)
            continue

        try:
            image = nib.load(str(file_path))
            shape = image.shape
            if any(size == 0 for size in shape):
                bad["emptyslice"].append((filename, shape))

            stats = quick_subsample_stats(file_path, step=4)
            if stats["empty"]:
                bad["emptyslice"].append((filename, shape))
            if not stats["finite_ok"]:
                bad["naninf"].append(filename)
            if not label_mode and float(stats["var"]) < 1e-8:
                bad["zerovar"].append(filename)
        except Exception:  # pylint: disable=broad-except
            bad["corrupt"].append(filename)

    def summary(tag: str) -> str:
        values = bad[tag]
        return f"{tag}: {len(values)}" + (f" (ej: {values[:max_list]})" if values else "")

    return bad, " | ".join(summary(key) for key in ["size0", "corrupt", "naninf", "zerovar", "emptyslice"])


def pad_center(arr: np.ndarray, target: Tuple[int, ...]) -> np.ndarray:
    pad_width = []
    for size, expected in zip(arr.shape, target):
        total = max(expected - size, 0)
        left = total // 2
        right = total - left
        pad_width.append((left, right))
    return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)


def bbox_from_mask(mask: np.ndarray):
    position = np.where(mask > 0)
    if len(position[0]) == 0:
        return None
    x_min, x_max = position[0].min(), position[0].max()
    y_min, y_max = position[1].min(), position[1].max()
    z_min, z_max = position[2].min(), position[2].max()
    return (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
