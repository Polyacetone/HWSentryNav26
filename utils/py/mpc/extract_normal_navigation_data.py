#!/usr/bin/env python3
"""Extract normal-navigation command/response pairs from ROS2 MCAP bags.

The input tree is scanned recursively and every ``*.mcap`` file is opened
read-only. March, May, and July 2026 interfaces are normalized into one NPZ
whose named arrays all share the command/response-pair timeline.

Normal navigation is defined at each chassis-status sample as:

* causal previous/ZOH command with mode 0 and age <= 0.12 s;
* causal executor/follower state FOLLOW (2) with age <= 0.12 s;
* chassis leg mode Mature (4);
* causal game progress 4 with age <= 3.0 s;
* the preceding and following five status samples satisfy the same conditions.

Only consecutive guarded status samples with 0.035 < dt < 0.075 s become
response pairs. Optional fields unavailable in an interface generation are
stored as NaN. Recovered bags are retained and explicitly marked.

Dependency and example:

    uv run --with mcap-ros2-support --with numpy \
      extract_normal_navigation_data.py \
      --input-dir /media/yuki/Disk \
      --output normal_navigation_data.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise RuntimeError(
        "mcap_ros2 is required. Run with: "
        "uv run --with mcap-ros2-support --with numpy <script>"
    ) from exc


COMMAND_TOPICS = {
    "/path_follower/chassis_cmd",
    "/nav_executor/chassis_cmd",
}
STATE_TOPICS = {
    "/path_follower/state",
    "/nav_executor/state",
}
CORE_TOPICS = COMMAND_TOPICS | STATE_TOPICS | {
    "/serial_bridge/chassis_status",
    "/serial_bridge/comp_stage",
}
AUXILIARY_TOPICS = {
    "/serial_bridge/imu_raw",
    "/small_glim/odometry",
    "/path_follower/debug/v_pred",
    "/path_follower/debug/w_pred",
    "/nav_executor/debug/v_pred",
    "/nav_executor/debug/w_pred",
}
ALL_TOPICS = CORE_TOPICS | AUXILIARY_TOPICS

GENERATION_NAMES = np.asarray(["unknown", "march", "may", "july"])


@dataclass(frozen=True)
class FilterConfig:
    max_command_age: float = 0.12
    max_state_age: float = 0.12
    max_stage_age: float = 3.0
    guard_samples: int = 5
    min_dt: float = 0.035
    max_dt: float = 0.075


@dataclass
class DecodedBag:
    arrays: dict[str, np.ndarray]
    schemas: dict[str, str]
    origin_ns: int
    decoded_message_count: int
    auxiliary_decode_error: str | None = None


def command_mode(msg: object) -> int:
    """Normalize current enum mode and the March boolean command interface."""
    if hasattr(msg, "mode"):
        return int(getattr(msg, "mode"))
    if bool(getattr(msg, "fast_spin", False)):
        return 2
    if bool(getattr(msg, "slow_spin", False)):
        return 1
    if bool(getattr(msg, "step_up_ahead", False)):
        return 3
    if bool(getattr(msg, "step_down_ahead", False)):
        return 6
    return 0


def motion_state(msg: object) -> int:
    return int(getattr(msg, "motion_state", getattr(msg, "state", -1)))


def append_row(rows: dict[str, list[list[float | int]]], key: str, row: list[float | int]) -> None:
    rows.setdefault(key, []).append(row)


def sorted_unique(rows: Iterable[Iterable[float | int]], width: int) -> np.ndarray:
    data = np.asarray(list(rows), dtype=np.float64)
    if data.size == 0:
        return np.empty((0, width), dtype=np.float64)
    data = data.reshape(-1, width)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    if len(data) > 1:
        # Overlapping chunks occur in recovered bags. The final message at a
        # duplicate log timestamp is the causal value retained for that topic.
        data = data[np.r_[data[:-1, 0] != data[1:, 0], True]]
    return data


def decode_mcap(path: Path, topics: set[str]) -> DecodedBag:
    rows: dict[str, list[list[float | int]]] = {}
    schemas: dict[str, str] = {}
    origin_ns: int | None = None
    message_count = 0

    for record in read_ros2_messages(path, topics=topics, log_time_order=True):
        topic = record.channel.topic
        msg = record.ros_msg
        log_time_ns = int(record.log_time_ns)
        if origin_ns is None:
            origin_ns = log_time_ns
        t = (log_time_ns - origin_ns) * 1e-9
        schemas[topic] = record.schema.name
        message_count += 1

        if topic in COMMAND_TOPICS:
            append_row(rows, "command", [t, float(msg.velocity), float(msg.omega), command_mode(msg)])
        elif topic == "/serial_bridge/chassis_status":
            append_row(rows, "status", [
                t,
                float(msg.velocity),
                float(msg.omega),
                float(getattr(msg, "leg_h", math.nan)),
                float(getattr(msg, "leg_psi", math.nan)),
                int(msg.leg_mode),
                float(getattr(msg, "curr_chassis_pwr", math.nan)),
                float(getattr(msg, "rfr_pwr_limit", math.nan)),
            ])
        elif topic == "/serial_bridge/comp_stage":
            append_row(rows, "stage", [
                t,
                int(msg.game_progress),
                float(getattr(msg, "stage_remain_time", math.nan)),
            ])
        elif topic in STATE_TOPICS:
            append_row(rows, "state", [t, motion_state(msg)])
        elif topic == "/serial_bridge/imu_raw":
            append_row(rows, "imu", [
                t,
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ])
        elif topic == "/small_glim/odometry":
            pose = msg.pose.pose
            twist = msg.twist.twist
            append_row(rows, "odom", [
                t,
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.linear.z),
                float(twist.angular.x),
                float(twist.angular.y),
                float(twist.angular.z),
            ])
        elif topic.endswith("/v_pred"):
            append_row(rows, "v_pred", [t, float(msg.data)])
        elif topic.endswith("/w_pred"):
            append_row(rows, "w_pred", [t, float(msg.data)])

    arrays = {
        "command": sorted_unique(rows.get("command", []), 4),
        "status": sorted_unique(rows.get("status", []), 8),
        "state": sorted_unique(rows.get("state", []), 2),
        "stage": sorted_unique(rows.get("stage", []), 3),
        "imu": sorted_unique(rows.get("imu", []), 7),
        "odom": sorted_unique(rows.get("odom", []), 14),
        "v_pred": sorted_unique(rows.get("v_pred", []), 2),
        "w_pred": sorted_unique(rows.get("w_pred", []), 2),
    }
    return DecodedBag(arrays, schemas, origin_ns or 0, message_count)


def decode_with_core_fallback(path: Path) -> DecodedBag:
    try:
        return decode_mcap(path, ALL_TOPICS)
    except Exception as exc:
        # Some historical bags contain an undecodable auxiliary schema while
        # their command/status/state/stage channels remain sound.
        core = decode_mcap(path, CORE_TOPICS)
        core.auxiliary_decode_error = f"{type(exc).__name__}: {exc}"
        return core


def previous_indices(table_t: np.ndarray, query_t: np.ndarray, max_age: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(table_t) == 0:
        return (
            np.zeros(len(query_t), dtype=np.int64),
            np.zeros(len(query_t), dtype=bool),
            np.full(len(query_t), np.nan),
        )
    index = np.searchsorted(table_t, query_t, side="right") - 1
    valid = index >= 0
    safe_index = np.maximum(index, 0)
    age = query_t - table_t[safe_index]
    valid &= (age >= -1e-9) & (age <= max_age)
    return safe_index, valid, age


def guarded_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = np.zeros(len(mask), dtype=bool)
    width = 2 * radius + 1
    if len(mask) < width:
        return result
    prefix = np.r_[0, np.cumsum(mask.astype(np.int64))]
    centers = np.arange(radius, len(mask) - radius)
    counts = prefix[centers + radius + 1] - prefix[centers - radius]
    result[centers] = counts == width
    return result


def interpolate_columns(table: np.ndarray, query_t: np.ndarray, columns: slice) -> np.ndarray:
    width = len(range(*columns.indices(table.shape[1]))) if table.ndim == 2 else 0
    result = np.full((len(query_t), width), np.nan)
    if len(table) < 2:
        return result
    for output_column, source_column in enumerate(range(*columns.indices(table.shape[1]))):
        result[:, output_column] = np.interp(
            query_t, table[:, 0], table[:, source_column], left=np.nan, right=np.nan
        )
    return result


def previous_values(table: np.ndarray, query_t: np.ndarray, max_age: float) -> tuple[np.ndarray, np.ndarray]:
    width = max(table.shape[1] - 1, 0)
    result = np.full((len(query_t), width), np.nan)
    index, valid, age = previous_indices(table[:, 0], query_t, max_age)
    if len(table):
        result[valid] = table[index[valid], 1:]
    return result, np.where(valid, age, np.nan)


def interval_peak_acceleration(imu: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    result = np.full(len(starts), np.nan)
    if len(imu) == 0:
        return result
    norm = np.linalg.norm(imu[:, 1:4], axis=1)
    for i, (start, end) in enumerate(zip(starts, ends)):
        lo = np.searchsorted(imu[:, 0], start, side="left")
        hi = np.searchsorted(imu[:, 0], end, side="right")
        if hi > lo:
            result[i] = float(np.max(norm[lo:hi]))
    return result


def infer_generation(schemas: dict[str, str], status: np.ndarray) -> int:
    if any(topic.startswith("/nav_executor/") for topic in schemas):
        return 3
    if any(topic.startswith("/path_follower/") for topic in schemas):
        if len(status) and np.any(np.isfinite(status[:, 3:5])):
            return 2
        return 1
    return 0


def empty_output_columns() -> dict[str, list[np.ndarray]]:
    return {}


def add_column(store: dict[str, list[np.ndarray]], name: str, values: np.ndarray) -> None:
    store.setdefault(name, []).append(np.asarray(values))


def add_matrix_columns(
    store: dict[str, list[np.ndarray]], names: list[str], values: np.ndarray
) -> None:
    if values.shape != (len(values), len(names)):
        raise ValueError(f"column shape mismatch for {names}: {values.shape}")
    for index, name in enumerate(names):
        add_column(store, name, values[:, index])


def build_pairs(
    decoded: DecodedBag,
    bag_index: int,
    recovered: bool,
    config: FilterConfig,
    segment_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    command = decoded.arrays["command"]
    status = decoded.arrays["status"]
    state = decoded.arrays["state"]
    stage = decoded.arrays["stage"]
    generation_code = infer_generation(decoded.schemas, status)

    if not all(len(table) for table in (command, status, state, stage)):
        return {}, {"status_samples": len(status), "normal_samples": 0, "pairs": 0}, segment_offset

    t = status[:, 0]
    command_index, command_valid, command_age = previous_indices(
        command[:, 0], t, config.max_command_age
    )
    state_index, state_valid, state_age = previous_indices(state[:, 0], t, config.max_state_age)
    stage_index, stage_valid, stage_age = previous_indices(stage[:, 0], t, config.max_stage_age)

    finite_core = (
        np.all(np.isfinite(status[:, 1:3]), axis=1)
        & np.all(np.isfinite(command[command_index, 1:3]), axis=1)
    )
    normal = (
        command_valid
        & state_valid
        & stage_valid
        & finite_core
        & (command[command_index, 3].astype(np.int64) == 0)
        & (state[state_index, 1].astype(np.int64) == 2)
        & (status[:, 5].astype(np.int64) == 4)
        & (stage[stage_index, 1].astype(np.int64) == 4)
    )
    guarded = guarded_mask(normal, config.guard_samples)
    dt = np.diff(t)
    pair_mask = (
        guarded[:-1]
        & guarded[1:]
        & (dt > config.min_dt)
        & (dt < config.max_dt)
    )
    starts = np.flatnonzero(pair_mask)
    ends = starts + 1
    count = len(starts)
    summary = {
        "status_samples": int(len(status)),
        "normal_samples": int(np.sum(normal)),
        "guarded_samples": int(np.sum(guarded)),
        "pairs": count,
    }
    if count == 0:
        return {}, summary, segment_offset

    # A new segment begins whenever selected pairs are not adjacent on the raw
    # status timeline. Segment IDs prevent later analyses crossing bag/gap edges.
    local_segment = np.cumsum(np.r_[True, starts[1:] != ends[:-1]]).astype(np.int64) - 1
    segment = local_segment + segment_offset
    next_segment_offset = int(segment[-1]) + 1
    response_t = t[ends]

    output: dict[str, np.ndarray] = {
        "bag_index": np.full(count, bag_index, dtype=np.int32),
        "generation_code": np.full(count, generation_code, dtype=np.uint8),
        "recovered": np.full(count, recovered, dtype=bool),
        "segment_index": segment.astype(np.int32),
        "status_index_start": starts.astype(np.int32),
        "t_start": t[starts],
        "t_response": response_t,
        "log_time_ns_start": decoded.origin_ns + np.rint(t[starts] * 1e9).astype(np.int64),
        "log_time_ns_response": decoded.origin_ns + np.rint(response_t * 1e9).astype(np.int64),
        "dt": dt[starts],
        "v_cmd": command[command_index[starts], 1],
        "w_cmd": command[command_index[starts], 2],
        "v_cmd_response": command[command_index[ends], 1],
        "w_cmd_response": command[command_index[ends], 2],
        "command_age": command_age[starts],
        "command_age_response": command_age[ends],
        "state_age": state_age[starts],
        "state_age_response": state_age[ends],
        "stage_age": stage_age[starts],
        "stage_age_response": stage_age[ends],
        "v_start": status[starts, 1],
        "w_start": status[starts, 2],
        "v_response": status[ends, 1],
        "w_response": status[ends, 2],
        "leg_h_start": status[starts, 3],
        "leg_psi_start": status[starts, 4],
        "leg_h_response": status[ends, 3],
        "leg_psi_response": status[ends, 4],
        "chassis_power_start": status[starts, 6],
        "chassis_power_response": status[ends, 6],
        "power_limit_start": status[starts, 7],
        "power_limit_response": status[ends, 7],
        "stage_remain_time": stage[stage_index[starts], 2],
        "stage_remain_time_response": stage[stage_index[ends], 2],
    }

    imu_values, imu_age = previous_values(decoded.arrays["imu"], response_t, 0.12)
    if imu_values.shape[1] == 6:
        add_matrix_columns(output_lists := empty_output_columns(), [
            "imu_accel_x", "imu_accel_y", "imu_accel_z",
            "imu_angular_x", "imu_angular_y", "imu_angular_z",
        ], imu_values)
        output.update({name: chunks[0] for name, chunks in output_lists.items()})
    output["imu_age"] = imu_age
    output["imu_accel_norm_peak"] = interval_peak_acceleration(
        decoded.arrays["imu"], t[starts], response_t
    )

    odom_values = interpolate_columns(decoded.arrays["odom"], response_t, slice(1, 14))
    odom_names = [
        "odom_position_x", "odom_position_y", "odom_position_z",
        "odom_orientation_x", "odom_orientation_y", "odom_orientation_z", "odom_orientation_w",
        "odom_linear_x", "odom_linear_y", "odom_linear_z",
        "odom_angular_x", "odom_angular_y", "odom_angular_z",
    ]
    output.update({name: odom_values[:, i] for i, name in enumerate(odom_names)})
    output["odom_horizontal_speed"] = np.hypot(odom_values[:, 7], odom_values[:, 8])

    v_pred, v_pred_age = previous_values(decoded.arrays["v_pred"], response_t, 0.12)
    w_pred, w_pred_age = previous_values(decoded.arrays["w_pred"], response_t, 0.12)
    output["logged_v_pred"] = v_pred[:, 0]
    output["logged_w_pred"] = w_pred[:, 0]
    output["logged_v_pred_age"] = v_pred_age
    output["logged_w_pred_age"] = w_pred_age
    return output, summary, next_segment_offset


def concatenate_columns(chunks: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values) for name, values in chunks.items()}


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **arrays,
                bag_names=np.asarray(metadata["bag_names"]),
                source_paths=np.asarray(metadata["source_paths"]),
                generation_names=GENERATION_NAMES,
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively extract guarded normal-navigation pairs from ROS2 MCAP bags."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("/media/yuki/Disk"),
        help="directory scanned recursively for *.mcap (default: /media/yuki/Disk)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("normal_navigation_data.npz"),
        help="combined output NPZ (default: ./normal_navigation_data.npz)",
    )
    parser.add_argument("--max-command-age", type=positive_float, default=0.12)
    parser.add_argument("--max-state-age", type=positive_float, default=0.12)
    parser.add_argument("--max-stage-age", type=positive_float, default=3.0)
    parser.add_argument("--guard-samples", type=nonnegative_int, default=5)
    parser.add_argument("--min-dt", type=positive_float, default=0.035)
    parser.add_argument("--max-dt", type=positive_float, default=0.075)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    if args.min_dt >= args.max_dt:
        print("ERROR: --min-dt must be less than --max-dt", file=sys.stderr)
        return 2

    files = sorted(path for path in args.input_dir.rglob("*.mcap") if path.is_file())
    if not files:
        print(f"ERROR: no MCAP files found below {args.input_dir}", file=sys.stderr)
        return 1

    config = FilterConfig(
        max_command_age=args.max_command_age,
        max_state_age=args.max_state_age,
        max_stage_age=args.max_stage_age,
        guard_samples=args.guard_samples,
        min_dt=args.min_dt,
        max_dt=args.max_dt,
    )
    started = time.monotonic()
    chunks: dict[str, list[np.ndarray]] = {}
    bag_names: list[str] = []
    source_paths: list[str] = []
    bag_summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    segment_offset = 0

    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] {path}", flush=True)
        bag_index = len(bag_names)
        bag_names.append(f"{path.parent.name}__{path.stem}")
        source_paths.append(str(path))
        bag_started = time.monotonic()
        try:
            decoded = decode_with_core_fallback(path)
            recovered = path.stem.lower().startswith("recovered")
            pair_columns, counts, segment_offset = build_pairs(
                decoded, bag_index, recovered, config, segment_offset
            )
            for name, values in pair_columns.items():
                add_column(chunks, name, values)
            generation_code = infer_generation(decoded.schemas, decoded.arrays["status"])
            summary = {
                "bag_index": bag_index,
                "bag_name": bag_names[-1],
                "source": str(path),
                "source_size": path.stat().st_size,
                "generation": str(GENERATION_NAMES[generation_code]),
                "recovered": recovered,
                "origin_ns": decoded.origin_ns,
                "decoded_message_count": decoded.decoded_message_count,
                "schemas": decoded.schemas,
                "raw_counts": {name: int(len(value)) for name, value in decoded.arrays.items()},
                **counts,
                "auxiliary_decode_error": decoded.auxiliary_decode_error,
                "elapsed_s": time.monotonic() - bag_started,
            }
            bag_summaries.append(summary)
            fallback = " (core-only fallback)" if decoded.auxiliary_decode_error else ""
            print(
                f"  {summary['generation']}: {counts['pairs']} pairs{fallback}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "bag_index": str(bag_index),
                "source": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"  ERROR: {failure['error']}", file=sys.stderr, flush=True)

    arrays = concatenate_columns(chunks)
    pair_count = len(arrays.get("dt", np.empty(0)))
    metadata: dict[str, object] = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "input_dir": str(args.input_dir.resolve()),
        "output": str(args.output.resolve()),
        "normal_navigation_definition": {
            "command_mode": 0,
            "motion_state_follow": 2,
            "leg_mode_mature": 4,
            "game_progress": 4,
            "max_command_age_s": config.max_command_age,
            "max_state_age_s": config.max_state_age,
            "max_stage_age_s": config.max_stage_age,
            "guard_samples_each_side": config.guard_samples,
            "min_dt_exclusive_s": config.min_dt,
            "max_dt_exclusive_s": config.max_dt,
            "command_alignment": "causal previous/ZOH",
        },
        "pair_count": pair_count,
        "segment_count": segment_offset,
        "discovered_mcap_count": len(files),
        "successful_mcap_count": len(bag_summaries),
        "failed_mcap_count": len(failures),
        "bag_names": bag_names,
        "source_paths": source_paths,
        "bags": bag_summaries,
        "failures": failures,
        "elapsed_s": time.monotonic() - started,
    }
    save_npz_atomic(args.output, arrays, metadata)

    print("Done")
    print(f"  output:   {args.output}")
    print(f"  MCAP:     {len(bag_summaries)}/{len(files)} successful")
    print(f"  pairs:    {pair_count}")
    print(f"  segments: {segment_offset}")
    if failures:
        print(f"  failures: {len(failures)} (see metadata_json in the NPZ)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
