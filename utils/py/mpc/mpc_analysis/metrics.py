"""Small metric helpers used by the LPV replay tools."""

from __future__ import annotations

import numpy as np


def residual_metrics(residual: np.ndarray, mask: np.ndarray, huber_delta: float = 0.08) -> dict[str, float | int]:
    values = np.asarray(residual, dtype=np.float64)
    values = values[np.asarray(mask, dtype=bool) & np.isfinite(values)]
    if not len(values):
        return {"n": 0, "bias": float("nan"), "rmse": float("nan"), "p95_abs": float("nan"), "huber": float("nan")}

    absolute = np.abs(values)
    huber = np.where(
        absolute <= huber_delta,
        0.5 * values * values,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return {
        "n": int(len(values)),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "huber": float(np.mean(huber)),
    }


def segment_acf(residual: np.ndarray, mask: np.ndarray, segment_id: np.ndarray, max_lag: int = 12) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(residual)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    output = np.full(max_lag + 1, np.nan)
    if not np.any(valid):
        return output

    centered = residual - np.mean(residual[valid])
    output[0] = 1.0
    for lag in range(1, max_lag + 1):
        pair = valid[:-lag] & valid[lag:] & (segment_id[:-lag] == segment_id[lag:])
        if not np.any(pair):
            continue
        left = centered[:-lag][pair]
        right = centered[lag:][pair]
        denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
        if denominator > 0.0:
            output[lag] = float(np.sum(left * right) / denominator)
    return output
