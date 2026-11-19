#!/usr/bin/env python3
"""Validate pretrained RF-DETR Small with RF-DETR's training-time evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from openpyxl import Workbook
from pytorch_lightning import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from rfdetr.config import RFDETRSmallConfig, TrainConfig  # noqa: E402
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback  # noqa: E402
from rfdetr.training.module_data import RFDETRDataModule  # noqa: E402
from rfdetr.training.module_model import RFDETRModelModule  # noqa: E402


COCO_TO_LOCAL_VEHICLE_LABEL = {3: 1, 6: 2, 8: 3}
LOCAL_CLASS_NAMES = {1: "car", 2: "bus", 3: "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-dets", type=int, default=500)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    return parser.parse_args()


class PretrainedVehicleValidationModule(RFDETRModelModule):
    """RF-DETR module that maps pretrained COCO vehicle labels to local IDs."""

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        samples, targets = batch
        outputs = self.model(samples)
        orig_sizes = torch.stack([target["orig_size"] for target in targets])
        raw_results = self.postprocess(outputs, orig_sizes)

        results: list[dict[str, torch.Tensor]] = []
        for raw in raw_results:
            labels = raw["labels"]
            keep = torch.zeros_like(labels, dtype=torch.bool)
            mapped = torch.zeros_like(labels)
            for coco_label, local_label in COCO_TO_LOCAL_VEHICLE_LABEL.items():
                selected = labels == coco_label
                keep |= selected
                mapped[selected] = local_label
            results.append(
                {
                    key: (mapped[keep] if key == "labels" else value[keep])
                    for key, value in raw.items()
                }
            )
        return {"results": results, "targets": targets}


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[str(key)] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def write_outputs(output_dir: Path, metrics: dict[str, float], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "rf-detr-small",
        "evaluation_protocol": "RF-DETR training validation",
        "dataset_root": str(args.dataset_root.resolve()),
        "device": args.device,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_detection_thresholds": [1, 10, args.max_dets],
        "map_backend": "faster_coco_eval",
        "f1_thresholds": "0.00..1.00, step 0.01; best macro-F1",
        "class_mapping": {"COCO car (3)": 1, "COCO bus (6)": 2, "COCO truck (8)": 3},
        "metrics": metrics,
    }
    (output_dir / "validation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(sorted(metrics.items()))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Метрики"
    sheet.append(["Модель", "Метрика", "Значение"])
    for key, value in sorted(metrics.items()):
        sheet.append(["rf-detr-small", key, value])
    settings = workbook.create_sheet("Параметры")
    settings.append(["Параметр", "Значение"])
    for key, value in payload.items():
        if key != "metrics":
            settings.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    workbook.save(output_dir / "rfdetr_small_training_protocol_validation.xlsx")


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = RFDETRSmallConfig(resolution=args.resolution, device=args.device)
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(args.dataset_root),
        output_dir=str(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        square_resize_div_64=True,
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=False,
        augmentation_backend="cpu",
        use_ema=False,
        compute_val_loss=False,
        eval_max_dets=args.max_dets,
        log_per_class_metrics=True,
        tensorboard=False,
        progress_bar="tqdm",
        accelerator=args.device,
    )

    module = PretrainedVehicleValidationModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule.setup("validate")

    callback = COCOEvalCallback(max_dets=args.max_dets, log_per_class_metrics=True)
    callback._class_names = ["car", "bus", "truck"]
    callback._cat_id_to_name = LOCAL_CLASS_NAMES.copy()

    trainer_kwargs: dict[str, Any] = {
        "accelerator": args.device,
        "devices": 1,
        "callbacks": [callback],
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": True,
        "precision": "16-mixed" if args.device == "mps" else "32-true",
    }
    if args.limit_val_batches > 0:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = Trainer(**trainer_kwargs)
    trainer.validate(module, datamodule=datamodule, verbose=True)
    metrics = scalar_metrics(dict(trainer.callback_metrics))
    write_outputs(args.output_dir, metrics, args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
