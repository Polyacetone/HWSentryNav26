"""Load the compact LPV pair dataset shared by the analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = (
    "segment_id",
    "train_mask",
    "test_mask",
    "dt",
    "v_cmd",
    "w_cmd",
    "v_cmd_response",
    "w_cmd_response",
    "v_start",
    "w_start",
    "v_response",
    "w_response",
    "leg_h_start",
    "leg_psi_start",
    "leg_h_response",
    "leg_psi_response",
)


@dataclass(frozen=True)
class PairDataset:
    arrays: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return len(self.arrays["dt"])

    def column(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"dataset does not contain column {name!r}") from exc

    @property
    def train(self) -> np.ndarray:
        return self.column("train_mask").astype(bool)

    @property
    def test(self) -> np.ndarray:
        return self.column("test_mask").astype(bool)

    def lpv_eligible(self) -> np.ndarray:
        values = np.column_stack([
            self.column(name)
            for name in (
                "dt", "v_cmd", "w_cmd", "v_cmd_response", "w_cmd_response",
                "v_start", "w_start", "v_response", "w_response",
                "leg_h_start", "leg_psi_start", "leg_h_response", "leg_psi_response",
            )
        ])
        return np.all(np.isfinite(values), axis=1)


def load_pair_dataset(path: Path) -> PairDataset:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_COLUMNS if name not in archive]
        if missing:
            raise ValueError(f"dataset is missing columns: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    size = len(arrays["dt"])
    bad_lengths = {name: len(values) for name, values in arrays.items() if len(values) != size}
    if bad_lengths:
        raise ValueError(f"dataset columns have inconsistent lengths: {bad_lengths}")

    train = arrays["train_mask"].astype(bool)
    test = arrays["test_mask"].astype(bool)
    if np.any(train & test) or not np.all(train | test):
        raise ValueError("train_mask and test_mask must be disjoint and exhaustive")
    if not np.any(train) or not np.any(test):
        raise ValueError("dataset must contain both train and test samples")

    segments = arrays["segment_id"].astype(np.int64)
    if size and np.any(np.diff(segments) < 0):
        raise ValueError("segment_id must be nondecreasing")
    return PairDataset(arrays)


def dilate_within_segments(event: np.ndarray, segment_id: np.ndarray, radius: int) -> np.ndarray:
    event = np.asarray(event, dtype=bool)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    if event.shape != segment_id.shape:
        raise ValueError("event and segment_id must have identical shapes")
    if radius <= 0 or not np.any(event):
        return event.copy()

    result = np.zeros(len(event), dtype=bool)
    boundaries = np.r_[0, np.flatnonzero(np.diff(segment_id) != 0) + 1, len(event)]
    kernel = np.ones(2 * radius + 1, dtype=np.int16)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        expanded = np.convolve(event[start:stop].astype(np.int16), kernel, mode="full")
        result[start:stop] = expanded[radius:radius + stop - start] > 0
    return result
