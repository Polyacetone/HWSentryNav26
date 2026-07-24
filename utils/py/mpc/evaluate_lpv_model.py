#!/usr/bin/env python3
"""Print train/test residual metrics for the current LPV model and observer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpc_analysis.dataset import load_pair_dataset
from mpc_analysis.metrics import residual_metrics
from mpc_analysis.model import load_model
from mpc_analysis.observer import model_observer_config, observer_duty, prepare_pairs, replay_observer, residual_masks


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "nav_executor/config/kinematic_model.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stress-threshold", type=float, default=0.12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stress_threshold <= 0.0:
        raise ValueError("--stress-threshold must be positive")

    dataset = load_pair_dataset(args.dataset)
    model = load_model(args.model)
    prepared = prepare_pairs(dataset, model)
    config = model_observer_config(model)
    shadow = replay_observer(prepared, config, hard_reset_enabled=False)
    production = replay_observer(prepared, config, hard_reset_enabled=True)
    masks = residual_masks(
        shadow,
        dataset.column("segment_id"),
        threshold=args.stress_threshold,
        dilation_radius=5,
    )

    print("split  n      v_rmse   v_bias   v_p95    yaw_rmse duty    resets")
    for name, split in (("train", dataset.train), ("test", dataset.test)):
        clean = split & masks["clean"]
        velocity = residual_metrics(shadow.velocity_innovation, clean)
        yaw = residual_metrics(shadow.angular_velocity_innovation, clean)
        duty = observer_duty(prepared, production, split)
        resets = int(np.sum(production.hard_reset & split))
        print(
            f"{name:<6} {velocity['n']:6d} "
            f"{velocity['rmse']:8.5f} {velocity['bias']:+8.5f} {velocity['p95_abs']:8.5f} "
            f"{yaw['rmse']:8.5f} {100.0 * duty:6.2f}% {resets:6d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
