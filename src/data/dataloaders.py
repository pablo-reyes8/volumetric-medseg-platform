"""Compatibility shim. The canonical data loaders now live in the top-level data package."""

from data.datasets import Hippocampus3DDataset, build_segmentation_dataloaders, split_volume_pairs
from data.preprocessing import minmax_normalize

__all__ = [
    "Hippocampus3DDataset",
    "build_segmentation_dataloaders",
    "minmax_normalize",
    "split_volume_pairs",
]
