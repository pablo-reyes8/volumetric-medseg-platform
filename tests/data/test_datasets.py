from pathlib import Path

import numpy as np

from data.datasets import Hippocampus3DDataset, build_segmentation_dataloaders, split_volume_pairs
from data.ingestion import pair_image_and_mask_files
from tests.utils import write_nifti


def test_pair_image_and_mask_files(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    volume = np.ones((6, 7, 5), dtype=np.float32)
    write_nifti(images_dir / "case_001.nii.gz", volume)
    write_nifti(labels_dir / "case_001.nii.gz", volume)

    pairs = pair_image_and_mask_files(images_dir, labels_dir)

    assert len(pairs) == 1
    assert pairs[0][0].name == "case_001.nii.gz"


def test_dataset_and_dataloader_bundle(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for index in range(4):
        volume = np.full((6, 7, 5), fill_value=index + 1, dtype=np.float32)
        mask = np.full((6, 7, 5), fill_value=index % 2, dtype=np.uint8)
        write_nifti(images_dir / f"case_{index:03d}.nii.gz", volume)
        write_nifti(labels_dir / f"case_{index:03d}.nii.gz", mask)

    pairs = pair_image_and_mask_files(images_dir, labels_dir)
    dataset = Hippocampus3DDataset(pairs)
    image_tensor, mask_tensor = dataset[0]
    loaders = build_segmentation_dataloaders(images_dir, labels_dir, batch_size=2, val_size=0.25)

    assert image_tensor.shape == (1, 6, 7, 5)
    assert mask_tensor.shape == (6, 7, 5)
    assert loaders["train_loader"] is not None
    assert len(loaders["train_pairs"]) + len(loaders["val_pairs"]) == 4
