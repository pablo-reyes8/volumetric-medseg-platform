"""Compatibility shim. Data ingestion now lives in the top-level data package."""

from data.ingestion import download_file, extract_archive, list_nifti_files, pair_image_and_mask_files, sha256_file
from data.versioning import build_dataset_manifest, load_dataset_manifest, save_dataset_manifest, update_dataset_registry

__all__ = [
    "build_dataset_manifest",
    "download_file",
    "extract_archive",
    "list_nifti_files",
    "load_dataset_manifest",
    "pair_image_and_mask_files",
    "save_dataset_manifest",
    "sha256_file",
    "update_dataset_registry",
]
