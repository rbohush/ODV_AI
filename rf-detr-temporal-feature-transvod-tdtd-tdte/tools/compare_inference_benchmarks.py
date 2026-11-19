from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and temporal inference benchmark JSON files.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--section", choices=["pure_model", "end_to_end"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metric_section(result: dict[str, Any], section: str | None) -> dict[str, Any]:
    if result.get("benchmark_type") == "both":
        if section is None:
            raise ValueError("--section is required when comparing files produced with --benchmark-mode both")
        return result[section]
    if section is not None and result.get("benchmark_type") != section:
        raise ValueError(f"Requested {section!r}, but file contains {result.get('benchmark_type')!r}")
    return result


def compare(baseline: dict[str, Any], temporal: dict[str, Any], section: str | None) -> dict[str, Any]:
    baseline_metrics = metric_section(baseline, section)
    temporal_metrics = metric_section(temporal, section)
    baseline_latency = float(baseline_metrics["latency_mean_ms"])
    temporal_latency = float(temporal_metrics["latency_mean_ms"])
    baseline_fps = float(baseline_metrics["fps"])
    temporal_fps = float(temporal_metrics["fps"])
    return {
        "benchmark_type": baseline_metrics["benchmark_type"],
        "dtype": baseline.get("dtype"),
        "baseline": {
            "latency_mean_ms": baseline_latency,
            "fps": baseline_fps,
            "input_shape": baseline_metrics["input_shape"],
        },
        "temporal": {
            "latency_mean_ms": temporal_latency,
            "fps": temporal_fps,
            "input_shape": temporal_metrics["input_shape"],
        },
        "latency_overhead_percent": ((temporal_latency - baseline_latency) / baseline_latency) * 100.0,
        "fps_drop_percent": ((baseline_fps - temporal_fps) / baseline_fps) * 100.0,
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal.read_text(encoding="utf-8"))
    result = compare(baseline, temporal, args.section)
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
