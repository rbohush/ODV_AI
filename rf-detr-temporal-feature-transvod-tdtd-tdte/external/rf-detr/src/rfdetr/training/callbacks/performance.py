# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Training performance logging callback."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.module import LightningModule


class TrainingPerformanceCallback(Callback):
    """Log per-step throughput and GPU memory during training.

    The callback records key-frame FPS for temporal models as one packed clip
    producing one key-frame prediction. It also logs raw-frame FPS as an auxiliary
    throughput metric, where a temporal clip with 3 reference frames counts as
    4 RGB frames.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        log_every_n_steps: int = 1,
        realtime_fps_target: float = 30.0,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.realtime_fps_target = float(realtime_fps_target)
        self.records: list[dict[str, Any]] = []
        self._start_wall: float | None = None
        self._last_batch_end_wall: float | None = None
        self._cuda_start_event: torch.cuda.Event | None = None
        self._fieldnames: list[str] | None = None
        self._csv_file: Any | None = None
        self._csv_writer: csv.DictWriter | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = (self.output_dir / "performance.csv").open("w", encoding="utf-8", newline="")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        now = time.perf_counter()
        self._start_wall = now
        if torch.cuda.is_available():
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero or self._start_wall is None:
            return
        if trainer.global_step % self.log_every_n_steps != 0 and batch_idx != 0:
            return

        wall_end = time.perf_counter()
        step_wall_ms = (wall_end - self._start_wall) * 1000.0
        data_wait_ms = (
            (self._start_wall - self._last_batch_end_wall) * 1000.0
            if self._last_batch_end_wall is not None
            else 0.0
        )
        self._last_batch_end_wall = wall_end

        device = pl_module.device
        gpu_step_ms = step_wall_ms
        if device.type == "cuda" and self._cuda_start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize(device)
            gpu_step_ms = float(self._cuda_start_event.elapsed_time(end_event))

        samples, targets = batch
        tensor = samples.tensors
        batch_size = int(tensor.shape[0])
        channels = int(tensor.shape[1])
        num_frames = max(1, channels // 3)
        key_fps_wall = batch_size * 1000.0 / step_wall_ms if step_wall_ms > 0 else 0.0
        key_fps_gpu = batch_size * 1000.0 / gpu_step_ms if gpu_step_ms > 0 else 0.0
        raw_frame_fps_wall = batch_size * num_frames * 1000.0 / step_wall_ms if step_wall_ms > 0 else 0.0
        raw_frame_fps_gpu = batch_size * num_frames * 1000.0 / gpu_step_ms if gpu_step_ms > 0 else 0.0

        allocated_gb = reserved_gb = peak_allocated_gb = peak_reserved_gb = 0.0
        if device.type == "cuda":
            gb = 1024**3
            allocated_gb = torch.cuda.memory_allocated(device) / gb
            reserved_gb = torch.cuda.memory_reserved(device) / gb
            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / gb
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / gb

        metrics = trainer.callback_metrics
        train_loss = metrics.get("train/loss")
        if torch.is_tensor(train_loss):
            train_loss_value: float | None = float(train_loss.detach().cpu())
        else:
            train_loss_value = None

        record = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "batch_idx": int(batch_idx),
            "input_shape": "x".join(str(dim) for dim in tensor.shape),
            "batch_size": batch_size,
            "num_frames_per_sample": num_frames,
            "num_key_frames": batch_size,
            "num_raw_frames": batch_size * num_frames,
            "num_targets": int(sum(len(target["labels"]) for target in targets)),
            "step_wall_ms": step_wall_ms,
            "gpu_step_ms": gpu_step_ms,
            "data_wait_ms": max(0.0, data_wait_ms),
            "ms_per_key_frame_wall": step_wall_ms / batch_size,
            "ms_per_key_frame_gpu": gpu_step_ms / batch_size,
            "key_fps_wall": key_fps_wall,
            "key_fps_gpu": key_fps_gpu,
            "raw_frame_fps_wall": raw_frame_fps_wall,
            "raw_frame_fps_gpu": raw_frame_fps_gpu,
            "realtime_target_fps": self.realtime_fps_target,
            "realtime_ratio_wall": key_fps_wall / self.realtime_fps_target if self.realtime_fps_target > 0 else 0.0,
            "realtime_ratio_gpu": key_fps_gpu / self.realtime_fps_target if self.realtime_fps_target > 0 else 0.0,
            "gpu_mem_allocated_gb": allocated_gb,
            "gpu_mem_reserved_gb": reserved_gb,
            "gpu_peak_allocated_gb": peak_allocated_gb,
            "gpu_peak_reserved_gb": peak_reserved_gb,
            "train_loss": train_loss_value,
        }
        self.records.append(record)
        self._write_csv_record(record)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero or not self.records:
            return
        summary = self._build_summary()
        (self.output_dir / "performance_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_plots()

    def _write_csv_record(self, record: dict[str, Any]) -> None:
        if self._csv_file is None:
            return
        if self._csv_writer is None:
            self._fieldnames = list(record.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def _build_summary(self) -> dict[str, Any]:
        def mean(key: str) -> float:
            return float(statistics.mean(float(record[key]) for record in self.records))

        def median(key: str) -> float:
            return float(statistics.median(float(record[key]) for record in self.records))

        def maximum(key: str) -> float:
            return float(max(float(record[key]) for record in self.records))

        return {
            "num_logged_steps": len(self.records),
            "avg_step_wall_ms": mean("step_wall_ms"),
            "median_step_wall_ms": median("step_wall_ms"),
            "max_step_wall_ms": maximum("step_wall_ms"),
            "avg_gpu_step_ms": mean("gpu_step_ms"),
            "median_gpu_step_ms": median("gpu_step_ms"),
            "max_gpu_step_ms": maximum("gpu_step_ms"),
            "avg_key_fps_wall": mean("key_fps_wall"),
            "avg_key_fps_gpu": mean("key_fps_gpu"),
            "avg_raw_frame_fps_wall": mean("raw_frame_fps_wall"),
            "avg_raw_frame_fps_gpu": mean("raw_frame_fps_gpu"),
            "avg_realtime_ratio_wall": mean("realtime_ratio_wall"),
            "avg_realtime_ratio_gpu": mean("realtime_ratio_gpu"),
            "avg_gpu_mem_allocated_gb": mean("gpu_mem_allocated_gb"),
            "avg_gpu_mem_reserved_gb": mean("gpu_mem_reserved_gb"),
            "max_gpu_peak_allocated_gb": maximum("gpu_peak_allocated_gb"),
            "max_gpu_peak_reserved_gb": maximum("gpu_peak_reserved_gb"),
            "realtime_target_fps": self.realtime_fps_target,
        }

    def _write_plots(self) -> None:
        try:
            mpl_config_dir = self.output_dir / ".matplotlib"
            mpl_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            (self.output_dir / "performance_plots_skipped.txt").write_text(str(exc), encoding="utf-8")
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        steps = [record["global_step"] for record in self.records]
        plot_specs = (
            ("key_fps_wall", "Key-frame FPS (wall)", "key_fps_wall.png"),
            ("gpu_step_ms", "GPU step time, ms", "gpu_step_ms.png"),
            ("step_wall_ms", "Wall step time, ms", "step_wall_ms.png"),
            ("gpu_mem_allocated_gb", "Allocated VRAM, GB", "gpu_mem_allocated_gb.png"),
            ("gpu_peak_allocated_gb", "Peak allocated VRAM, GB", "gpu_peak_allocated_gb.png"),
            ("realtime_ratio_wall", "Realtime ratio vs target FPS", "realtime_ratio_wall.png"),
        )
        for key, title, filename in plot_specs:
            plt.figure(figsize=(9, 4.8))
            plt.plot(steps, [record[key] for record in self.records], linewidth=1.6)
            if key == "realtime_ratio_wall":
                plt.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
            plt.title(title)
            plt.xlabel("global step")
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=160)
            plt.close()
