import numpy as np
import pytest


@pytest.fixture
def sample_volume() -> np.ndarray:
    values = np.arange(6 * 7 * 5, dtype=np.float32).reshape(6, 7, 5)
    return values / values.max()


@pytest.fixture
def sample_mask() -> np.ndarray:
    mask = np.zeros((6, 7, 5), dtype=np.uint8)
    mask[1:4, 2:6, 1:4] = 1
    mask[4:, 1:3, 2:] = 2
    return mask
