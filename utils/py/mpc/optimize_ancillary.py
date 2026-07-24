#!/usr/bin/env python3
"""Generate one conservative ancillary-feedback A/B candidate from the train split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from mpc_analysis.ancillary import AncillaryConfig, persistent_bias, simulate_field_errors, velocity_rmse
from mpc_analysis.dataset import load_pair_dataset
from mpc_analysis.model import load_model
from mpc_analysis.observer import ObserverConfig, prepare_pairs, replay_observer, residual_masks


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "nav_executor/config/kinematic_model.yaml"
DEFAULT_CONTROLLER = ROOT / "nav_executor/config/path_executor.yaml"

MIN_PERSISTENT_IMPROVEMENT = 0.08
MIN_CLEAN_IMPROVEMENT = 0.005
MAX_STRESS_DEGRADATION = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--observer-params", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ancillary_params.yaml"))
    return parser.parse_args()


def nested_mapping(path: Path, keys: tuple[str, ...]) -> dict[str, object]:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing {'.'.join(keys)} in {path}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping at {'.'.join(keys)} in {path}")
    return value


def load_current_ancillary(path: Path) -> AncillaryConfig:
    values = nested_mapping(path, ("/**", "ros__parameters", "mpc", "follow", "ancillary_feedback"))
    return AncillaryConfig(
        velocity_error_gain=float(values["velocity_error_gain"]),
        command_error_gain=float(values["command_error_gain"]),
        velocity_error_reanchor_threshold=float(values["velocity_error_reanchor_threshold"]),
        velocity_command_margin=float(values["velocity_command_margin"]),
        velocity_command_rate_margin=float(values["velocity_command_rate_margin"]),
    )


def load_observer(model, path: Path | None) -> ObserverConfig:
    if path is None:
        return ObserverConfig(
            model.observer.velocity_gain,
            model.observer.correction_clip,
            model.observer.reset_threshold,
        )
    values = nested_mapping(path, ("/**", "ros__parameters", "kinematic_model"))
    return ObserverConfig(
        float(values["obs_lv"]),
        float(values["obs_v_correction_clip"]),
        float(values["obs_v_reset_threshold"]),
    )


def first_value(values: tuple[float, ...], predicate) -> float | None:
    return next((value for value in values if predicate(value)), None)


def candidate_grid(reanchor: float) -> list[AncillaryConfig]:
    command_margins = (0.02, 0.03, 0.05, 0.08, 0.10)
    rate_margins = (0.05, 0.08, 0.10, 0.15, 0.20, 0.25)
    result = []
    for velocity_gain in (0.1, 0.3, 0.5, 0.8):
        required = velocity_gain * reanchor
        for command_gain in (2.0, 3.0, 4.0, 6.0):
            command_margin = first_value(command_margins, lambda value: command_gain * value > required)
            rate_margin = first_value(rate_margins, lambda value: value >= required)
            if command_margin is None or rate_margin is None:
                continue
            config = AncillaryConfig(
                velocity_gain,
                command_gain,
                reanchor,
                command_margin,
                rate_margin,
            )
            if config.authority_valid():
                result.append(config)
    return result


def ratio(value: float, baseline: float) -> float:
    return value / baseline if math.isfinite(baseline) and baseline > 0.0 else math.inf


def main() -> int:
    args = parse_args()
    dataset = load_pair_dataset(args.dataset)
    model = load_model(args.model)
    prepared = prepare_pairs(dataset, model)
    observer = load_observer(model, args.observer_params)
    replay = replay_observer(prepared, observer, hard_reset_enabled=False)
    masks = residual_masks(replay, dataset.column("segment_id"), threshold=0.12, dilation_radius=5)
    valid = masks["eligible"]
    split_masks = {
        "train_clean": dataset.train & masks["clean"],
        "train_stress": dataset.train & masks["stress"],
        "test_clean": dataset.test & masks["clean"],
        "test_stress": dataset.test & masks["stress"],
    }

    current = load_current_ancillary(args.controller_config)
    disabled = AncillaryConfig(0.0, 0.0, current.velocity_error_reanchor_threshold, 0.0, 0.0)
    configs = {tuple(config.serializable().values()): config for config in [disabled, current, *candidate_grid(current.velocity_error_reanchor_threshold)]}

    rows = []
    for config in configs.values():
        simulation = simulate_field_errors(prepared, replay.velocity_innovation, valid, config)
        row = {
            "config": config,
            "persistent": persistent_bias(model, config),
        }
        for name, mask in split_masks.items():
            row[name] = velocity_rmse(simulation, mask)
        rows.append(row)

    baseline = next(row for row in rows if row["config"] == disabled)
    baseline_persistent = float(baseline["persistent"])
    for row in rows:
        row["persistent_ratio"] = ratio(float(row["persistent"]), baseline_persistent)
        for name in split_masks:
            row[f"{name}_ratio"] = ratio(
                float(row[name]),
                float(baseline[name]),
            )
        row["score"] = (
            float(row["persistent_ratio"])
            + 0.5 * float(row["train_clean_ratio"])
            + 5.0 * max(0.0, float(row["train_stress_ratio"]) - 1.0)
        )

    candidates = [row for row in rows if row["config"] not in (disabled, current)]
    constrained = [
        row for row in candidates
        if 1.0 - float(row["persistent_ratio"]) >= MIN_PERSISTENT_IMPROVEMENT
        and 1.0 - float(row["train_clean_ratio"]) >= MIN_CLEAN_IMPROVEMENT
        and float(row["train_stress_ratio"]) - 1.0 <= MAX_STRESS_DEGRADATION
    ]
    if constrained:
        constrained.sort(key=lambda row: (
            row["config"].velocity_command_margin,
            row["config"].velocity_command_rate_margin,
            row["config"].velocity_error_gain,
            row["config"].command_error_gain,
            float(row["score"]),
        ))
        selected = constrained[0]
        selection_status = "passed_train_constraints"
    else:
        candidates.sort(key=lambda row: float(row["score"]))
        selected = candidates[0]
        selection_status = "fallback_no_candidate_passed_train_constraints"

    test_clean_improvement = 1.0 - float(selected["test_clean_ratio"])
    test_stress_change = float(selected["test_stress_ratio"]) - 1.0
    test_passed = (
        test_clean_improvement >= MIN_CLEAN_IMPROVEMENT
        and test_stress_change <= MAX_STRESS_DEGRADATION
    )
    test_status = "passed" if test_passed else "failed"
    selected_config = selected["config"]
    assert isinstance(selected_config, AncillaryConfig)

    patch = {
        "/**": {
            "ros__parameters": {
                "mpc": {
                    "follow": {
                        "ancillary_feedback": selected_config.serializable(),
                    }
                }
            }
        }
    }
    warning = (
        "# Counterfactual A/B candidate only; existing bags contain no ancillary closed-loop data.\n"
        f"# selection_status: {selection_status}, test_status: {test_status}\n"
        f"# test_clean_improvement: {100.0 * test_clean_improvement:+.2f}%, "
        f"test_stress_change: {100.0 * test_stress_change:+.2f}%\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(warning + yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")

    ranked = sorted(candidates, key=lambda row: float(row["score"]))
    displayed = [selected, *(row for row in ranked if row is not selected)][:5]
    print("Kv   Kcmd cmd_margin rate_margin train_clean train_stress test_clean test_stress")
    for row in displayed:
        config = row["config"]
        assert isinstance(config, AncillaryConfig)
        marker = "*" if row is selected else " "
        print(
            f"{marker}{config.velocity_error_gain:3.1f} {config.command_error_gain:5.1f} "
            f"{config.velocity_command_margin:10.3f} {config.velocity_command_rate_margin:11.3f} "
            f"{100.0 * (1.0 - float(row['train_clean_ratio'])):+10.2f}% "
            f"{100.0 * (float(row['train_stress_ratio']) - 1.0):+11.2f}% "
            f"{100.0 * (1.0 - float(row['test_clean_ratio'])):+9.2f}% "
            f"{100.0 * (float(row['test_stress_ratio']) - 1.0):+10.2f}%"
        )
    print(f"saved: {args.output}")
    print(f"test status: {test_status}; real-robot validation is still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
