# dataset/jepa_frames_dataset.py
# [JEPA] Dataset for offline ViZDoom frames stored as .npy (N, C, H, W).

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class JEPAFramesDataset(Dataset):
    """
    Offline dataset for JEPA pretraining.

    Each .npy file is expected to have shape (N, C, H, W), where:
        C = frame_stack * 3  (e.g., 12 for 4 RGB frames),
        values are uint8 in [0, 255].

    This dataset returns normalized float32 tensors in [0, 1]:
        x: (C, H, W)
    """

    def __init__(self, data_paths: Sequence[str | Path]) -> None:
        super().__init__()
        paths: List[Path] = [Path(p) for p in data_paths]
        arrays: List[np.ndarray] = []

        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"[JEPA] Missing data file: {p}")
            arr = np.load(p)
            if arr.ndim != 4:
                raise ValueError(f"[JEPA] Expected (N, C, H, W) in {p}, got {arr.shape}")
            arrays.append(arr)

        if not arrays:
            raise ValueError("[JEPA] No data arrays loaded.")

        self.frames = np.concatenate(arrays, axis=0)  # (N_total, C, H, W)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.frames[idx]  # (C, H, W), uint8
        x_t = torch.from_numpy(x).float() / 255.0
        return x_t

class JEPAFramesTemporalDataset(Dataset):
    """
    Temporal JEPA: return (frame_t, frame_{t+delta})
    Assuming .npy shape = (T, C, H, W), along the time axis
    """
    def __init__(self, frame_paths, delta: int = 1):
        arrays = [np.load(p, mmap_mode="r") for p in frame_paths]
        self.frames = np.concatenate(arrays, axis=0)  # (N, C, H, W)
        self.delta = delta
        # if last delta frame cannot be feature, leave it
        self._length = self.frames.shape[0] - self.delta

    def __len__(self) -> int:
        return max(self._length, 0)

    def __getitem__(self, idx: int):
        x_t = self.frames[idx]                 # (C,H,W)
        x_tp = self.frames[idx + self.delta]   # (C,H,W)
        # return torch.Tensor, uint8 remain unchanged, model will turn it to float
        return torch.from_numpy(x_t), torch.from_numpy(x_tp)