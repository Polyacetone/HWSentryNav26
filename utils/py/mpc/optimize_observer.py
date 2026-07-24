#!/usr/bin/env python3
"""Select observer gain/clip on the NPZ train split and report the test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from mpc_analysis.dataset import load_pair_dataset
from mpc_analysis.metrics import residual_metrics, segment_acf
from mpc_analysis.model import load_model
from mpc_analysis.observer import (
    ObserverConfig,
    model_observer_config,
    observer_duty,
    prepare_pairs,
    replay_observer,
    residual_masks,
)


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "nav_executor/config/kinematic_model.yaml"


def comma_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("observer_params.yaml"))
    parser.add_argument("--gains", type=comma_floats, default=comma_floats("0,1,2,3,4,4.484294986367658,5,6,7,8"))
    parser.add_argument("--clips", type=comma_floats, default=comma_floats("0.05,0.08,0.10,0.12,0.16,0.20"))
    parser.add_argument("--minimum-duty", type=float, default=0.95)
    return parser.parse_args()


def evaluate(prepared, config: ObserverConfig, clean: np.ndarray) -> dict[str, float | int | ObserverConfig]:
    shadow = replay_observer(prepared, config, hard_reset_enabled=False)
    production = replay_observer(prepared, config, hard_reset_enabled=True)
    segment_id = prepared.dataset.column("segment_id")
    row: dict[str, float | int | ObserverConfig] = {"config": config}
    for name, split in (("train", prepared.dataset.train), ("test", prepared.dataset.test)):
        selected = split & clean & shadow.prediction_available
        metrics = residual_metrics(shadow.velocity_innovation, selected)
        acf = segment_acf(shadow.velocity_innovation, selected, segment_id)
        finite_acf = acf[1:][np.isfinite(acf[1:])]
        row[f"{name}_rmse"] = float(metrics["rmse"])
        row[f"{name}_acf"] = float(np.sqrt(np.mean(finite_acf * finite_acf))) if len(finite_acf) else math.inf
        row[f"{name}_duty"] = observer_duty(prepared, production, split)
        row[f"{name}_resets"] = int(np.sum(production.hard_reset & split))
    row["score"] = float(row["train_rmse"]) + 0.01 * float(row["train_acf"])
    return row


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_duty <= 1.0:
        raise ValueError("--minimum-duty must be in (0, 1]")

    dataset = load_pair_dataset(args.dataset)
    model = load_model(args.model)
    prepared = prepare_pairs(dataset, model)
    current = model_observer_config(model)
    reference = replay_observer(prepared, current, hard_reset_enabled=False)
    clean = residual_masks(reference, dataset.column("segment_id"), threshold=0.12, dilation_radius=5)["clean"]

    configs = {
        (float(gain), float(clip)): ObserverConfig(float(gain), float(clip), current.reset_threshold)
        for gain in args.gains
        for clip in args.clips
        if 0.0 < clip < current.reset_threshold
    }
    configs[(current.velocity_gain, current.correction_clip)] = current
    rows = [evaluate(prepared, config, clean) for config in configs.values()]
    feasible = [row for row in rows if float(row["train_duty"]) >= args.minimum_duty]
    if not feasible:
        raise RuntimeError("no observer candidate satisfies the minimum train duty")
    feasible.sort(key=lambda row: (float(row["score"]), float(row["train_rmse"])))
    selected = feasible[0]
    current_row = next(row for row in rows if row["config"] == current)
    selected_config = selected["config"]
    assert isinstance(selected_config, ObserverConfig)

    test_improvement = 1.0 - float(selected["test_rmse"]) / float(current_row["test_rmse"])
    test_status = "improved" if test_improvement > 0.0 else "not_improved"
    patch = {
        "/**": {
            "ros__parameters": {
                "kinematic_model": {
                    "obs_lv": selected_config.velocity_gain,
                    "obs_v_correction_clip": selected_config.correction_clip,
                    "obs_v_reset_threshold": selected_config.reset_threshold,
                }
            }
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warning = (
        "# Train-selected observer candidate; test split was not used for ranking.\n"
        f"# test_status: {test_status}, rmse_improvement: {100.0 * test_improvement:+.2f}%\n"
    )
    args.output.write_text(warning + yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")

    print("gain   clip   score    train_rmse test_rmse train_duty")
    for row in feasible[:5]:
        config = row["config"]
        assert isinstance(config, ObserverConfig)
        print(
            f"{config.velocity_gain:5.2f} {config.correction_clip:6.3f} {float(row['score']):8.5f} "
            f"{float(row['train_rmse']):10.5f} {float(row['test_rmse']):9.5f} "
            f"{100.0 * float(row['train_duty']):9.2f}%"
        )
    print(f"saved: {args.output}")
    print(f"test RMSE improvement versus current: {100.0 * test_improvement:+.2f}% ({test_status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
