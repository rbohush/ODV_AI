from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = REPO_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from rfdetr._namespace import _namespace_from_configs  # noqa: E402
from rfdetr.config import RFDETRSmallConfig, TrainConfig  # noqa: E402
from rfdetr.datasets import build_dataset  # noqa: E402
from rfdetr.models import build_criterion_from_config, build_model_from_config  # noqa: E402
from rfdetr.utilities.tensors import make_collate_fn  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    name: str
    temporal_num_ref_frames: int
    temporal_fusion_layers: int = 1
    temporal_tdte_layers: int = 0
    temporal_tdtd_layers: int = 0
    temporal_ref_frame_mode: str = "surrounding"
    temporal_key_frame_position: str = "first"


RUN_SPECS = (
    RunSpec(name="baseline", temporal_num_ref_frames=0),
    RunSpec(name="temporal_tqe", temporal_num_ref_frames=3),
    RunSpec(
        name="temporal_tdte_tdtd_tqe",
        temporal_num_ref_frames=3,
        temporal_tdte_layers=1,
        temporal_tdtd_layers=1,
        temporal_key_frame_position="center",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short RF-DETR training smoke benchmark with per-step FPS/VRAM logs and plots."
    )
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "car_dataset_temporal"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "result" / "training_smoke_small"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-max-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_mem_allocated_gb": 0.0,
            "gpu_mem_reserved_gb": 0.0,
            "gpu_peak_allocated_gb": 0.0,
            "gpu_peak_reserved_gb": 0.0,
        }
    gb = 1024**3
    return {
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated(device) / gb,
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved(device) / gb,
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / gb,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / gb,
    }


def _move_batch_to_device(batch: tuple[Any, Any], device: torch.device) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    samples, targets = batch
    non_blocking = device.type == "cuda"
    samples = samples.to(device, non_blocking=non_blocking)
    targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in target.items()} for target in targets]
    return samples, targets


def _weighted_loss(loss_dict: dict[str, torch.Tensor], weight_dict: dict[str, float]) -> torch.Tensor:
    losses = [loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict]
    if not losses:
        raise RuntimeError("Criterion returned no weighted losses.")
    return sum(losses)


def _build_configs(args: argparse.Namespace, spec: RunSpec) -> tuple[RFDETRSmallConfig, TrainConfig]:
    model_config = RFDETRSmallConfig(
        pretrain_weights=None,
        num_classes=args.num_classes,
        resolution=args.resolution,
        amp=False,
        compile=False,
        temporal_num_ref_frames=spec.temporal_num_ref_frames,
        temporal_fusion_layers=spec.temporal_fusion_layers,
        temporal_tdte_layers=spec.temporal_tdte_layers,
        temporal_tdtd_layers=spec.temporal_tdtd_layers,
        temporal_key_frame_position=spec.temporal_key_frame_position,
        device=args.device,
    )
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        epochs=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_max_norm=args.clip_max_norm,
        num_workers=args.num_workers,
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=False,
        augmentation_backend="cpu",
        use_ema=False,
        tensorboard=False,
        temporal_ref_frame_mode=spec.temporal_ref_frame_mode,
        temporal_key_frame_position=spec.temporal_key_frame_position,
        seed=args.seed,
    )
    return model_config, train_config


def _build_loader(model_config: RFDETRSmallConfig, train_config: TrainConfig) -> DataLoader:
    namespace = _namespace_from_configs(model_config, train_config)
    dataset = build_dataset("train", namespace, model_config.resolution)
    block_size = model_config.patch_size * model_config.num_windows
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collate_fn(block_size=block_size),
    )


def run_one(args: argparse.Namespace, spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    _set_deterministic_seed(args.seed)
    model_config, train_config = _build_configs(args, spec)
    loader = _build_loader(model_config, train_config)
    data_iter = iter(loader)

    model = build_model_from_config(model_config, train_config).to(device)
    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion = criterion.to(device)
    model.train()
    criterion.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        fused=False,
    )

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    num_frames = spec.temporal_num_ref_frames + 1
    records: list[dict[str, Any]] = []
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{spec.name}.jsonl"

    total_steps = args.warmup_steps + args.steps
    with log_path.open("w", encoding="utf-8") as log_file:
        for local_step in range(total_steps):
            data_started = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            data_time_ms = (time.perf_counter() - data_started) * 1000.0

            input_shape = list(batch[0].tensors.shape)
            transfer_started = time.perf_counter()
            samples, targets = _move_batch_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_time_ms = (time.perf_counter() - transfer_started) * 1000.0

            optimizer.zero_grad(set_to_none=True)
            wall_started = time.perf_counter()
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = _weighted_loss(loss_dict, criterion.weight_dict)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_max_norm)
            optimizer.step()

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize(device)
                compute_gpu_ms = float(start_event.elapsed_time(end_event))
            else:
                compute_gpu_ms = (time.perf_counter() - wall_started) * 1000.0
            compute_wall_ms = (time.perf_counter() - wall_started) * 1000.0
            step_wall_ms = data_time_ms + transfer_time_ms + compute_wall_ms

            key_fps = (args.batch_size * 1000.0) / compute_gpu_ms if compute_gpu_ms > 0 else math.nan
            raw_frame_fps = (args.batch_size * num_frames * 1000.0) / compute_gpu_ms if compute_gpu_ms > 0 else math.nan
            record = {
                "run": spec.name,
                "step": local_step,
                "measured": local_step >= args.warmup_steps,
                "variant": "small",
                "device": str(device),
                "input_shape": input_shape,
                "num_frames_per_sample": num_frames,
                "batch_size": args.batch_size,
                "data_time_ms": data_time_ms,
                "transfer_time_ms": transfer_time_ms,
                "compute_gpu_ms": compute_gpu_ms,
                "compute_wall_ms": compute_wall_ms,
                "step_wall_ms": step_wall_ms,
                "ms_per_key_frame": compute_gpu_ms / args.batch_size,
                "ms_per_raw_frame": compute_gpu_ms / (args.batch_size * num_frames),
                "key_fps": key_fps,
                "raw_frame_fps": raw_frame_fps,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "num_targets": int(sum(len(target["labels"]) for target in targets)),
                "params_m": params_m,
                **{
                    f"loss_{key}": float(value.detach().cpu())
                    for key, value in loss_dict.items()
                    if torch.is_tensor(value) and value.numel() == 1
                },
                **_memory_stats(device),
            }
            log_file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            log_file.flush()
            records.append(record)
            print(
                f"[{spec.name}] step {local_step + 1}/{total_steps} "
                f"measured={record['measured']} loss={record['loss']:.4f} "
                f"compute={compute_gpu_ms:.1f} ms key_fps={key_fps:.2f} "
                f"peak_vram={record['gpu_peak_allocated_gb']:.2f} GB",
                flush=True,
            )

    measured = [record for record in records if record["measured"]]
    summary = {
        "run": spec.name,
        "variant": "small",
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "temporal_num_ref_frames": spec.temporal_num_ref_frames,
        "temporal_fusion_layers": spec.temporal_fusion_layers,
        "temporal_tdte_layers": spec.temporal_tdte_layers,
        "temporal_tdtd_layers": spec.temporal_tdtd_layers,
        "temporal_ref_frame_mode": spec.temporal_ref_frame_mode,
        "temporal_key_frame_position": spec.temporal_key_frame_position,
        "input_shape": measured[-1]["input_shape"] if measured else records[-1]["input_shape"],
        "params_m": params_m,
        "loss_last": records[-1]["loss"],
        "compute_ms_mean": statistics.mean(record["compute_gpu_ms"] for record in measured),
        "compute_ms_median": statistics.median(record["compute_gpu_ms"] for record in measured),
        "step_wall_ms_mean": statistics.mean(record["step_wall_ms"] for record in measured),
        "data_time_ms_mean": statistics.mean(record["data_time_ms"] for record in measured),
        "transfer_time_ms_mean": statistics.mean(record["transfer_time_ms"] for record in measured),
        "key_fps_mean": statistics.mean(record["key_fps"] for record in measured),
        "raw_frame_fps_mean": statistics.mean(record["raw_frame_fps"] for record in measured),
        "peak_vram_gb": max(record["gpu_peak_allocated_gb"] for record in records),
        "log_path": str(log_path),
    }
    return summary


def write_plots(output_dir: Path, summaries: list[dict[str, Any]], records_by_run: dict[str, list[dict[str, Any]]]) -> None:
    try:
        mpl_config_dir = output_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(f"Plot generation skipped: matplotlib is unavailable ({exc}).", flush=True)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [summary["run"] for summary in summaries]

    bar_specs = (
        ("compute_ms_mean", "Mean train compute time, ms", "compute_ms_mean.png"),
        ("key_fps_mean", "Mean key-frame FPS", "key_fps_mean.png"),
        ("peak_vram_gb", "Peak allocated VRAM, GB", "peak_vram_gb.png"),
    )
    for key, title, filename in bar_specs:
        plt.figure(figsize=(8, 4.5))
        plt.bar(names, [summary[key] for summary in summaries], color=["#4C78A8", "#F58518", "#54A24B"])
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    for name, records in records_by_run.items():
        measured = [record for record in records if record["measured"]]
        plt.plot(
            [record["step"] for record in measured],
            [record["compute_gpu_ms"] for record in measured],
            marker="o",
            label=name,
        )
    plt.title("Per-step train compute time")
    plt.xlabel("Step")
    plt.ylabel("ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "per_step_compute_ms.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    records_by_run: dict[str, list[dict[str, Any]]] = {}
    for spec in RUN_SPECS:
        summary = run_one(args, spec, output_dir)
        summaries.append(summary)
        records_by_run[spec.name] = [
            json.loads(line)
            for line in Path(summary["log_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    baseline = next(summary for summary in summaries if summary["run"] == "baseline")
    for summary in summaries:
        if summary["run"] == "baseline":
            summary["latency_overhead_vs_baseline_percent"] = 0.0
            summary["key_fps_drop_vs_baseline_percent"] = 0.0
            continue
        summary["latency_overhead_vs_baseline_percent"] = (
            (summary["compute_ms_mean"] - baseline["compute_ms_mean"]) / baseline["compute_ms_mean"] * 100.0
        )
        summary["key_fps_drop_vs_baseline_percent"] = (
            (baseline["key_fps_mean"] - summary["key_fps_mean"]) / baseline["key_fps_mean"] * 100.0
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.plot:
        write_plots(output_dir, summaries, records_by_run)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
