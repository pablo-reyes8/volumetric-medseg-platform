import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


VALID_NIFTI_SUFFIXES = (".nii", ".nii.gz")


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def list_nifti_files(path: Path | str) -> List[Path]:
    directory = Path(path)
    files = [
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.name.endswith(VALID_NIFTI_SUFFIXES) and not file_path.name.startswith("._")
    ]
    return sorted(files, key=lambda file_path: strip_nii_suffix(file_path.name))


def pair_image_and_mask_files(images_dir: Path | str, labels_dir: Path | str) -> List[Tuple[Path, Path]]:
    images = {strip_nii_suffix(path.name): path for path in list_nifti_files(images_dir)}
    labels = {strip_nii_suffix(path.name): path for path in list_nifti_files(labels_dir)}

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise ValueError(
            "No se pudieron emparejar imagenes y mascaras. "
            f"missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )

    return [(images[key], labels[key]) for key in sorted(images)]


def download_file(url: str, destination: Path | str, overwrite: bool = False, timeout: int = 120) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() and not overwrite:
        return destination_path

    request = urllib.request.Request(url, headers={"User-Agent": "unet3d-medseg/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, destination_path.open("wb") as output_file:
        output_file.write(response.read())

    return destination_path


def extract_archive(archive_path: Path | str, destination: Path | str, overwrite: bool = False) -> Path:
    archive = Path(archive_path)
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(target_dir)
        return target_dir

    suffixes = archive.suffixes
    if suffixes[-1] == ".tar" or suffixes[-2:] in [[".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]]:
        with tarfile.open(archive, mode="r:*") as tar_file:
            tar_file.extractall(target_dir)
        return target_dir

    raise ValueError(f"Formato de archivo no soportado para extraccion: {archive.name}")


def sha256_file(path: Path | str, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
