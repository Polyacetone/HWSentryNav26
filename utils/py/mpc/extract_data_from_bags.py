#!/usr/bin/env python3
"""Extract LPV command/response pairs from every usable MCAP below a directory."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages


MAX_COMMAND_AGE = 0.12
MAX_STATE_AGE = 0.12
MAX_STAGE_AGE = 3.0
GUARD_SAMPLES = 5
MIN_DT = 0.035
MAX_DT = 0.075

STATUS_FIELDS = {"velocity", "omega", "leg_h", "leg_psi", "leg_mode"}
COMMAND_MARKERS = {"mode", "fast_spin", "slow_spin", "step_up_ahead", "step_down_ahead"}
STATE_TYPES = {"FollowerState", "NavExecutorState", "NavExecutorDiag"}
FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SkipBag(RuntimeError):
    pass


def schema_fields(data: bytes) -> set[str]:
    text = data.decode(errors="ignore").split("=" * 80, 1)[0]
    fields: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("MSG:") or "=" in line:
            continue
        parts = line.replace(";", " ").split()
        if len(parts) >= 2 and FIELD_NAME.fullmatch(parts[1]):
            fields.add(parts[1])
    return fields


def classify_topic(topics: dict[str, set[str]], topic: str, type_name: str, schema_data: bytes) -> None:
    fields = schema_fields(schema_data)
    short_type = type_name.rsplit("/", 1)[-1]
    topic_tail = topic.rstrip("/").rsplit("/", 1)[-1]

    if short_type == "ChassisStatus" or {"velocity", "omega", "leg_mode"} <= fields:
        topics["status"].add(topic)
    elif short_type == "ChassisCmd" or ({"velocity", "omega"} <= fields and fields & COMMAND_MARKERS):
        topics["command"].add(topic)
    elif short_type == "CompStage" or "game_progress" in fields:
        topics["stage"].add(topic)
    elif short_type in STATE_TYPES or "motion_state" in fields or ("state" in fields and topic_tail == "state"):
        topics["state"].add(topic)


def discover_topics(path: Path) -> dict[str, set[str]]:
    topics = {name: set() for name in ("command", "status", "state", "stage")}
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is not None:
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                if schema is not None:
                    classify_topic(topics, channel.topic, schema.name, schema.data)
        else:
            seen_channels: set[int] = set()
            for schema, channel, _ in reader.iter_messages(log_time_order=False):
                if channel.id in seen_channels or schema is None:
                    continue
                seen_channels.add(channel.id)
                classify_topic(topics, channel.topic, schema.name, schema.data)
                if all(topics.values()):
                    break

    missing = [name for name, values in topics.items() if not values]
    if missing:
        raise SkipBag(f"missing logical streams: {', '.join(missing)}")
    return topics


def command_mode(message: object) -> int:
    if hasattr(message, "mode"):
        return int(getattr(message, "mode"))
    if bool(getattr(message, "fast_spin", False)):
        return 2
    if bool(getattr(message, "slow_spin", False)):
        return 1
    if bool(getattr(message, "step_up_ahead", False)):
        return 3
    if bool(getattr(message, "step_down_ahead", False)):
        return 6
    return 0


def motion_state(message: object) -> int:
    return int(getattr(message, "motion_state", getattr(message, "state", -1)))


def sorted_unique(rows: list[list[float]], width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=np.float64)
    values = np.asarray(rows, dtype=np.float64).reshape(-1, width)
    values = values[np.argsort(values[:, 0], kind="stable")]
    if len(values) > 1:
        values = values[np.r_[values[:-1, 0] != values[1:, 0], True]]
    return values


def decode_bag(path: Path, topics: dict[str, set[str]]) -> dict[str, np.ndarray]:
    rows: dict[str, list[list[float]]] = {name: [] for name in topics}
    selected_topics = set().union(*topics.values())
    origin_ns: int | None = None

    for record in read_ros2_messages(path, topics=selected_topics, log_time_order=True):
        if origin_ns is None:
            origin_ns = int(record.log_time_ns)
        time_s = (int(record.log_time_ns) - origin_ns) * 1e-9
        topic = record.channel.topic
        message = record.ros_msg

        if topic in topics["status"]:
            rows["status"].append([
                time_s,
                float(getattr(message, "velocity", math.nan)),
                float(getattr(message, "omega", math.nan)),
                float(getattr(message, "leg_h", math.nan)),
                float(getattr(message, "leg_psi", math.nan)),
                float(getattr(message, "leg_mode", -1)),
            ])
        elif topic in topics["command"]:
            rows["command"].append([
                time_s,
                float(getattr(message, "velocity", math.nan)),
                float(getattr(message, "omega", math.nan)),
                float(command_mode(message)),
            ])
        elif topic in topics["stage"]:
            rows["stage"].append([time_s, float(getattr(message, "game_progress", -1))])
        elif topic in topics["state"]:
            rows["state"].append([time_s, float(motion_state(message))])

    return {
        "command": sorted_unique(rows["command"], 4),
        "status": sorted_unique(rows["status"], 6),
        "state": sorted_unique(rows["state"], 2),
        "stage": sorted_unique(rows["stage"], 2),
    }


def previous_indices(table_time: np.ndarray, query_time: np.ndarray, max_age: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(table_time, query_time, side="right") - 1
    valid = index >= 0
    safe_index = np.maximum(index, 0)
    age = query_time - table_time[safe_index]
    valid &= (age >= -1e-9) & (age <= max_age)
    return safe_index, valid


def guarded_mask(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=bool)
    width = 2 * GUARD_SAMPLES + 1
    if len(mask) < width:
        return result
    prefix = np.r_[0, np.cumsum(mask.astype(np.int64))]
    centers = np.arange(GUARD_SAMPLES, len(mask) - GUARD_SAMPLES)
    count = prefix[centers + GUARD_SAMPLES + 1] - prefix[centers - GUARD_SAMPLES]
    result[centers] = count == width
    return result


def build_pairs(tables: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    command = tables["command"]
    status = tables["status"]
    state = tables["state"]
    stage = tables["stage"]
    if not all(len(table) for table in (command, status, state, stage)):
        raise SkipBag("one or more logical streams contain no messages")
    if not np.any(np.all(np.isfinite(status[:, 3:5]), axis=1)):
        raise SkipBag("chassis status lacks leg_h/leg_psi required by the LPV model")

    time_s = status[:, 0]
    command_index, command_valid = previous_indices(command[:, 0], time_s, MAX_COMMAND_AGE)
    state_index, state_valid = previous_indices(state[:, 0], time_s, MAX_STATE_AGE)
    stage_index, stage_valid = previous_indices(stage[:, 0], time_s, MAX_STAGE_AGE)

    finite = (
        np.all(np.isfinite(status[:, 1:5]), axis=1)
        & np.all(np.isfinite(command[command_index, 1:3]), axis=1)
    )
    normal = (
        command_valid
        & state_valid
        & stage_valid
        & finite
        & (command[command_index, 3].astype(np.int64) == 0)
        & (state[state_index, 1].astype(np.int64) == 2)
        & (status[:, 5].astype(np.int64) == 4)
        & (stage[stage_index, 1].astype(np.int64) == 4)
    )
    guarded = guarded_mask(normal)
    dt = np.diff(time_s)
    starts = np.flatnonzero(
        guarded[:-1] & guarded[1:] & (dt > MIN_DT) & (dt < MAX_DT)
    )
    if not len(starts):
        raise SkipBag("no guarded normal-navigation pairs")
    responses = starts + 1

    local_segment = np.cumsum(np.r_[True, starts[1:] != responses[:-1]]).astype(np.int64) - 1
    arrays = {
        "segment_id": local_segment.astype(np.int32),
        "dt": dt[starts],
        "v_cmd": command[command_index[starts], 1],
        "w_cmd": command[command_index[starts], 2],
        "v_cmd_response": command[command_index[responses], 1],
        "w_cmd_response": command[command_index[responses], 2],
        "v_start": status[starts, 1],
        "w_start": status[starts, 2],
        "v_response": status[responses, 1],
        "w_response": status[responses, 2],
        "leg_h_start": status[starts, 3],
        "leg_psi_start": status[starts, 4],
        "leg_h_response": status[responses, 3],
        "leg_psi_response": status[responses, 4],
    }
    return arrays, int(local_segment[-1]) + 1


def ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/media/yuki/Disk"))
    parser.add_argument("--output", type=Path, default=Path("mpc_pairs.npz"))
    parser.add_argument("--train-ratio", type=ratio, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    paths = sorted(path for path in args.input_dir.rglob("*.mcap") if path.is_file())
    if not paths:
        print(f"ERROR: no MCAP files found below {args.input_dir}", file=sys.stderr)
        return 1

    bags: list[dict[str, np.ndarray]] = []
    segment_offset = 0
    skipped = 0
    for index, path in enumerate(paths, 1):
        try:
            arrays, segment_count = build_pairs(decode_bag(path, discover_topics(path)))
            arrays["segment_id"] = arrays["segment_id"].astype(np.int64) + segment_offset
            segment_offset += segment_count
            bags.append(arrays)
            print(f"[{index}/{len(paths)}] {path}: {len(arrays['dt'])} pairs")
        except Exception as exc:
            skipped += 1
            print(f"[{index}/{len(paths)}] {path}: skipped ({exc})")

    if len(bags) < 2:
        print("ERROR: at least two usable bags are required for a train/test split", file=sys.stderr)
        return 1

    generator = np.random.default_rng(args.seed)
    shuffled = generator.permutation(len(bags))
    train_bag_count = min(max(int(args.train_ratio * len(bags)), 1), len(bags) - 1)
    train_bags = set(int(value) for value in shuffled[:train_bag_count])

    chunks: dict[str, list[np.ndarray]] = {}
    for bag_index, arrays in enumerate(bags):
        count = len(arrays["dt"])
        arrays["train_mask"] = np.full(count, bag_index in train_bags, dtype=bool)
        arrays["test_mask"] = ~arrays["train_mask"]
        for name, values in arrays.items():
            chunks.setdefault(name, []).append(values)
    combined = {name: np.concatenate(values) for name, values in chunks.items()}
    save_npz(args.output, combined)

    print(f"saved: {args.output}")
    print(f"usable/skipped bags: {len(bags)}/{skipped}")
    print(f"train bags/pairs: {train_bag_count}/{int(np.sum(combined['train_mask']))}")
    print(f"test bags/pairs: {len(bags) - train_bag_count}/{int(np.sum(combined['test_mask']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
