import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def write_nifti(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))
    return path


def nifti_bytes(data: np.ndarray) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        write_nifti(path, data)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)
