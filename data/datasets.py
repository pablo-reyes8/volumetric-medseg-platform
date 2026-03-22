from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data.ingestion import pair_image_and_mask_files
from data.preprocessing import load_nifti_volume, minmax_normalize


@dataclass(frozen=True)
class VolumePair:
    image_path: Path
    mask_path: Path


class Hippocampus3DDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], norm: str = "minmax", clip=(1, 99), dtype_img=np.float32):
        self.pairs = [(Path(image_path), Path(mask_path)) for image_path, mask_path in pairs]
        self.norm = norm
        self.clip = clip
        self.dtype_img = dtype_img

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image, _, _ = load_nifti_volume(image_path, dtype=self.dtype_img)
        mask, _, _ = load_nifti_volume(mask_path, dtype=np.int16)

        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch {image.shape} vs {mask.shape} en {image_path.name}")

        if self.norm == "minmax":
            image = minmax_normalize(image, clip=self.clip)
        elif self.norm is None:
            image = image.astype(np.float32)
        else:
            raise ValueError(f"norm '{self.norm}' no soportada. Usa 'minmax' o None.")

        image_tensor = torch.from_numpy(image[None, ...]).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


def split_volume_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    pair_list = [(Path(image_path), Path(mask_path)) for image_path, mask_path in pairs]
    if len(pair_list) < 2:
        return pair_list, []
    train_pairs, val_pairs = train_test_split(pair_list, test_size=val_size, random_state=seed, shuffle=True)
    return list(train_pairs), list(val_pairs)


def build_segmentation_dataloaders(
    images_dir: Path | str,
    labels_dir: Path | str,
    batch_size: int = 1,
    val_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    norm: str = "minmax",
    clip=(1, 99),
):
    pairs = pair_image_and_mask_files(images_dir, labels_dir)
    train_pairs, val_pairs = split_volume_pairs(pairs, val_size=val_size, seed=seed)
    train_dataset = Hippocampus3DDataset(train_pairs, norm=norm, clip=clip)
    val_dataset = Hippocampus3DDataset(val_pairs, norm=norm, clip=clip) if val_pairs else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
    }
