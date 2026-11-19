#!/usr/bin/env python3
"""Validate pretrained RF-DETR Small and a temporal checkpoint on one COCO split.

The evaluator maps the pretrained COCO vehicle classes (car, bus, truck) to the
three local dataset categories, evaluates both models with identical settings,
and writes COCO metrics plus fixed-threshold precision/recall/F1 to Excel.

Inference is resumable. Each processed image is appended to a JSONL cache in
the output directory; rerunning the same command skips completed images.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

COCO_VEHICLE_LABELS = {3: "car", 6: "bus", 8: "truck"}
TEMPORAL_LOCAL_LABELS = {1: "car", 2: "bus", 3: "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--shape", type=int, default=640)
    parser.add_argument("--coco-score-threshold", type=float, default=0.001)
    parser.add_argument("--operating-threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["checkpoint", "previous", "surrounding", "duplicate"],
        default="checkpoint",
    )
    parser.add_argument("--temporal-ref-frame-step", type=int, default=0)
    parser.add_argument(
        "--models",
        choices=["both", "rf-detr-small", "temporal"],
        default="both",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=["car", "bus", "truck"],
        default=["car", "bus", "truck"],
        help="Dataset classes included in predictions, GT, and all metrics.",
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Discard resumable JSONL prediction caches and start inference again.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent / "training_config.json"
    return load_json(path) if path.is_file() else {}


def checkpoint_temporal_settings(checkpoint: Path) -> dict[str, Any]:
    payload = load_training_config(checkpoint)
    model_cfg = payload.get("model_config", {})
    train_cfg = payload.get("train_config", {})
    notes = train_cfg.get("notes", {})
    return {
        "num_classes": int(model_cfg.get("num_classes", payload.get("num_classes", 3))),
        "resolution": int(model_cfg.get("resolution", 640)),
        "backbone_lora": bool(model_cfg.get("backbone_lora", notes.get("backbone_lora", True))),
        "temporal_num_ref_frames": int(model_cfg.get("temporal_num_ref_frames", notes.get("temporal_num_ref_frames", 3))),
        "temporal_fusion_layers": int(model_cfg.get("temporal_fusion_layers", notes.get("temporal_fusion_layers", 1))),
        "temporal_key_frame_position": str(model_cfg.get("temporal_key_frame_position", notes.get("temporal_key_frame_position", "center"))),
        "temporal_tdte_layers": int(model_cfg.get("temporal_tdte_layers", notes.get("temporal_tdte_layers", 1))),
        "temporal_tdtd_layers": int(model_cfg.get("temporal_tdtd_layers", notes.get("temporal_tdtd_layers", 1))),
        "temporal_dropout": float(model_cfg.get("temporal_dropout", 0.0)),
        "ref_frame_mode": str(train_cfg.get("temporal_ref_frame_mode", notes.get("temporal_ref_frame_mode", "surrounding"))),
        "ref_frame_step": int(train_cfg.get("temporal_ref_frame_step", notes.get("temporal_ref_frame_step", 2))),
    }


def load_baseline(device: torch.device) -> Any:
    from rfdetr import RFDETRSmall

    wrapper = RFDETRSmall()
    wrapper.model.device = device
    wrapper.model.model.to(device).eval()
    return wrapper


def load_temporal(checkpoint: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    from rfdetr import RFDETRSmall, from_checkpoint

    settings = checkpoint_temporal_settings(checkpoint)
    if checkpoint.suffix.lower() == ".ckpt":
        wrapper = RFDETRSmall(
            num_classes=settings["num_classes"],
            resolution=settings["resolution"],
            backbone_lora=settings["backbone_lora"],
            temporal_num_ref_frames=settings["temporal_num_ref_frames"],
            temporal_fusion_layers=settings["temporal_fusion_layers"],
            temporal_key_frame_position=settings["temporal_key_frame_position"],
            temporal_tdte_layers=settings["temporal_tdte_layers"],
            temporal_tdtd_layers=settings["temporal_tdtd_layers"],
            temporal_dropout=settings["temporal_dropout"],
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise KeyError(f"Lightning checkpoint has no state_dict: {checkpoint}")
        state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
        incompatible = wrapper.model.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Temporal checkpoint is incompatible with the reconstructed model: "
                f"missing={incompatible.missing_keys[:10]}, unexpected={incompatible.unexpected_keys[:10]}"
            )
        settings["checkpoint_epoch"] = int(payload.get("epoch", -1))
        settings["checkpoint_global_step"] = int(payload.get("global_step", -1))
    else:
        wrapper = from_checkpoint(checkpoint)
        settings["checkpoint_epoch"] = None
        settings["checkpoint_global_step"] = None
    wrapper.model.device = device
    wrapper.model.model.to(device).eval()
    return wrapper, settings


def temporal_offsets(num_refs: int, mode: str, step: int) -> list[int]:
    if num_refs <= 0:
        return []
    if mode == "duplicate":
        return [0] * num_refs
    if mode == "previous":
        return [-(index + 1) * step for index in range(num_refs)]
    if mode == "surrounding":
        offsets: list[int] = []
        radius = 1
        while len(offsets) < num_refs:
            offsets.extend([-radius * step, radius * step])
            radius += 1
        return offsets[:num_refs]
    raise ValueError(f"Unsupported temporal reference mode: {mode}")


def temporal_clip_offsets(num_refs: int, mode: str, step: int, key_frame_position: str) -> list[int]:
    """Return offsets in the exact channel order used by TemporalCocoDetection."""
    if num_refs <= 0:
        return [0]
    if key_frame_position == "first":
        return [0, *temporal_offsets(num_refs, mode, step)]
    if key_frame_position != "center":
        raise ValueError(f"Unsupported temporal key-frame position: {key_frame_position}")
    if mode == "duplicate":
        return [0] * (num_refs + 1)
    key_index = (num_refs + 1) // 2
    return [step * (frame_index - key_index) for frame_index in range(num_refs + 1)]


def build_video_index(images: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[int, tuple[str, int]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        video_id = str(image.get("video_id", image["file_name"]))
        groups[video_id].append(image)
    positions: dict[int, tuple[str, int]] = {}
    for video_id, sequence in groups.items():
        sequence.sort(key=lambda item: (int(item.get("frame_id", 0)), int(item["id"])))
        for index, image in enumerate(sequence):
            positions[int(image["id"])] = (video_id, index)
    return dict(groups), positions


def select_clip_images(
    image: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    positions: dict[int, tuple[str, int]],
    clip_offsets: list[int],
) -> list[dict[str, Any]]:
    video_id, position = positions[int(image["id"])]
    sequence = groups[video_id]
    return [sequence[min(max(position + offset, 0), len(sequence) - 1)] for offset in clip_offsets]


class TensorCache:
    def __init__(self, image_dir: Path, shape: int, device: torch.device, max_items: int = 12) -> None:
        self.image_dir = image_dir
        self.shape = shape
        self.device = device
        self.max_items = max_items
        self._data: dict[str, torch.Tensor] = {}
        self._order: list[str] = []

    def get(self, file_name: str) -> torch.Tensor:
        if file_name in self._data:
            self._order.remove(file_name)
            self._order.append(file_name)
            return self._data[file_name]
        path = self.image_dir / file_name
        with Image.open(path) as image:
            tensor = F.to_tensor(image.convert("RGB")).to(self.device)
        tensor = F.resize(tensor, [self.shape, self.shape], antialias=True)
        tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._data[file_name] = tensor
        self._order.append(file_name)
        while len(self._order) > self.max_items:
            oldest = self._order.pop(0)
            del self._data[oldest]
        return tensor

    def clear(self) -> None:
        self._data.clear()
        self._order.clear()


def local_category_maps(
    coco: COCO, selected_classes: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    all_id_to_name = {
        int(cat_id): str(cat["name"]).lower() for cat_id, cat in coco.cats.items()
    }
    all_name_to_id = {name: cat_id for cat_id, name in all_id_to_name.items()}
    selected = set(selected_classes)
    missing = sorted(selected - set(all_name_to_id))
    if missing:
        raise ValueError(f"Validation categories do not contain requested classes: {missing}")
    id_to_name = {
        cat_id: name for cat_id, name in all_id_to_name.items() if name in selected
    }
    name_to_id = {name: cat_id for cat_id, name in id_to_name.items()}
    return name_to_id, id_to_name


def mapped_category_id(model_name: str, raw_label: int, name_to_id: dict[str, int]) -> int | None:
    label_names = COCO_VEHICLE_LABELS if model_name == "rf-detr-small" else TEMPORAL_LOCAL_LABELS
    class_name = label_names.get(int(raw_label))
    return name_to_id.get(class_name) if class_name is not None else None


def read_prediction_cache(path: Path) -> tuple[set[int], list[dict[str, Any]]]:
    completed: set[int] = set()
    predictions: list[dict[str, Any]] = []
    if not path.is_file():
        return completed, predictions
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete cache line {line_number} in {path}", flush=True)
                break
            completed.add(int(record["image_id"]))
            predictions.extend(record.get("predictions", []))
    return completed, predictions


def run_inference(
    model_name: str,
    wrapper: Any,
    images: list[dict[str, Any]],
    image_dir: Path,
    device: torch.device,
    shape: int,
    score_threshold: float,
    max_detections: int,
    name_to_id: dict[str, int],
    cache_path: Path,
    groups: dict[str, list[dict[str, Any]]],
    positions: dict[int, tuple[str, int]],
    clip_offsets: list[int],
    overwrite: bool,
) -> list[dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and cache_path.exists():
        cache_path.unlink()
    completed, predictions = read_prediction_cache(cache_path)
    print(
        f"{model_name}: images={len(images)}, already_cached={len(completed)}, "
        f"clip_offsets={clip_offsets}, device={device}",
        flush=True,
    )
    tensor_cache = TensorCache(image_dir, shape, device, max_items=max(4, len(clip_offsets) * 4))
    with cache_path.open("a", encoding="utf-8") as cache_handle, torch.inference_mode():
        for index, image in enumerate(images, start=1):
            image_id = int(image["id"])
            if image_id in completed:
                continue
            clip = select_clip_images(image, groups, positions, clip_offsets)
            tensor = torch.cat([tensor_cache.get(item["file_name"]) for item in clip], dim=0).unsqueeze(0)
            outputs = wrapper.model.model(tensor)
            target_sizes = torch.tensor([[int(image["height"]), int(image["width"])]], device=device)
            result = wrapper.model.postprocess(outputs, target_sizes=target_sizes)[0]
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            order = np.argsort(-scores)[:max_detections]
            image_predictions: list[dict[str, Any]] = []
            for pred_index in order:
                score = float(scores[pred_index])
                if score < score_threshold:
                    continue
                category_id = mapped_category_id(model_name, int(labels[pred_index]), name_to_id)
                if category_id is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in boxes[pred_index]]
                image_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": score,
                    }
                )
            record = {"image_id": image_id, "predictions": image_predictions}
            cache_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 10 == 0:
                cache_handle.flush()
            predictions.extend(image_predictions)
            if index % 50 == 0 or index == len(images):
                print(f"{model_name}: processed {index}/{len(images)}", flush=True)
            if device.type == "mps" and index % 50 == 0:
                torch.mps.empty_cache()
    tensor_cache.clear()
    return predictions


def xywh_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def fixed_threshold_metrics(
    coco: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    score_threshold: float,
    iou_threshold: float,
    id_to_name: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if float(pred["score"]) >= score_threshold:
            predictions_by_image[int(pred["image_id"])].append(pred)

    class_counts = {cat_id: {"gt": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0} for cat_id in id_to_name}
    for image_id in image_ids:
        annotations = [ann for ann in coco.imgToAnns.get(image_id, []) if not bool(ann.get("iscrowd", 0))]
        by_category_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_category_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in annotations:
            if int(ann["category_id"]) in class_counts:
                by_category_gt[int(ann["category_id"])].append(ann)
        for pred in predictions_by_image.get(image_id, []):
            if int(pred["category_id"]) in class_counts:
                by_category_pred[int(pred["category_id"])].append(pred)
        for category_id in class_counts:
            gt_items = by_category_gt.get(category_id, [])
            pred_items = sorted(by_category_pred.get(category_id, []), key=lambda item: float(item["score"]), reverse=True)
            class_counts[category_id]["gt"] += len(gt_items)
            class_counts[category_id]["predictions"] += len(pred_items)
            used_gt: set[int] = set()
            for pred in pred_items:
                best_gt = None
                best_iou = 0.0
                for gt_index, gt in enumerate(gt_items):
                    if gt_index in used_gt:
                        continue
                    value = xywh_iou(pred["bbox"], gt["bbox"])
                    if value > best_iou:
                        best_gt = gt_index
                        best_iou = value
                if best_gt is not None and best_iou >= iou_threshold:
                    used_gt.add(best_gt)
                    class_counts[category_id]["tp"] += 1
                else:
                    class_counts[category_id]["fp"] += 1
            class_counts[category_id]["fn"] += len(gt_items) - len(used_gt)

    per_class: list[dict[str, Any]] = []
    for category_id, counts in class_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "category_id": category_id,
                "class": id_to_name[category_id],
                **counts,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    totals = {key: sum(int(row[key]) for row in per_class) for key in ("gt", "predictions", "tp", "fp", "fn")}
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    totals["precision"] = precision
    totals["recall"] = recall
    totals["F1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return totals, sorted(per_class, key=lambda item: int(item["category_id"]))


def safe_mean(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def run_coco_eval(
    coco: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
    max_detections: int,
) -> tuple[dict[str, Any], dict[int, dict[str, float | None]], str]:
    if not predictions:
        raise RuntimeError("No predictions were retained for COCO evaluation")
    coco_dt = coco.loadRes(predictions)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.params.maxDets = [1, 10, max_detections]
    output = StringIO()
    with redirect_stdout(output):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    text = output.getvalue()
    print(text, end="", flush=True)
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou_thresholds = evaluator.params.iouThrs
    iou50 = int(np.argmin(np.abs(iou_thresholds - 0.50)))
    iou75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))
    summary = {
        "mAP_50_95": safe_mean(precision[:, :, :, 0, -1]),
        "mAP_50": safe_mean(precision[iou50, :, :, 0, -1]),
        "mAP_75": safe_mean(precision[iou75, :, :, 0, -1]),
        "mAP_small": safe_mean(precision[:, :, :, 1, -1]),
        "mAP_medium": safe_mean(precision[:, :, :, 2, -1]),
        "mAP_large": safe_mean(precision[:, :, :, 3, -1]),
        "mAR_1": safe_mean(recall[:, :, 0, 0]),
        "mAR_10": safe_mean(recall[:, :, 0, 1]),
        f"mAR_{max_detections}": safe_mean(recall[:, :, 0, -1]),
        "mAR_small": safe_mean(recall[:, :, 1, -1]),
        "mAR_medium": safe_mean(recall[:, :, 2, -1]),
        "mAR_large": safe_mean(recall[:, :, 3, -1]),
    }
    per_class: dict[int, dict[str, float | None]] = {}
    for class_index, category_id in enumerate(category_ids):
        per_class[category_id] = {
            "AP_50_95": safe_mean(precision[:, :, class_index, 0, -1]),
            "AP_50": safe_mean(precision[iou50, :, class_index, 0, -1]),
            "AP_75": safe_mean(precision[iou75, :, class_index, 0, -1]),
            f"AR_{max_detections}": safe_mean(recall[:, class_index, 0, -1]),
        }
    return summary, per_class, text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_excel(
    path: Path,
    summary_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    config_rows: list[tuple[str, Any]],
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to write Excel: pip install openpyxl") from exc

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Сравнение моделей"
    class_sheet = workbook.create_sheet("Метрики по классам")
    config_sheet = workbook.create_sheet("Параметры валидации")
    thin = Side(style="thin", color="808080")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill = PatternFill("solid", fgColor="D9EAF7")

    def populate(sheet: Any, rows: list[dict[str, Any]], table_name: str) -> None:
        headers = list(rows[0])
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header) for header in headers])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in sheet.iter_rows():
            for cell in row:
                cell.border = border
                if cell.row > 1:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
        sheet.freeze_panes = "B2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, header in enumerate(headers, start=1):
            width = max(len(str(header)) + 2, 13)
            sheet.column_dimensions[get_column_letter(index)].width = min(width, 28)
        table = Table(displayName=table_name, ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        sheet.add_table(table)

    populate(summary_sheet, summary_rows, "ModelComparison")
    populate(class_sheet, per_class_rows, "PerClassMetrics")
    config_sheet.append(["Параметр", "Значение"])
    for key, value in config_rows:
        config_sheet.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    for row in config_sheet.iter_rows():
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for cell in config_sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = fill
    config_sheet.column_dimensions["A"].width = 34
    config_sheet.column_dimensions["B"].width = 100
    config_sheet.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def release_model(wrapper: Any, device: torch.device) -> None:
    del wrapper
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    annotation_path = args.dataset_root / "annotations" / "instances_val2017.json"
    image_dir = args.dataset_root / "val2017"
    if not annotation_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"Expected {annotation_path} and {image_dir}")
    if not args.temporal_checkpoint.is_file():
        raise FileNotFoundError(args.temporal_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(annotation_path))
    images = sorted(coco.loadImgs(coco.getImgIds()), key=lambda item: int(item["id"]))
    if args.max_images > 0:
        images = images[: args.max_images]
    image_ids = [int(image["id"]) for image in images]
    name_to_id, id_to_name = local_category_maps(coco, args.classes)
    category_ids = sorted(id_to_name)
    groups, positions = build_video_index(coco.loadImgs(coco.getImgIds()))
    temporal_settings = checkpoint_temporal_settings(args.temporal_checkpoint)
    temporal_mode = (
        temporal_settings["ref_frame_mode"]
        if args.temporal_ref_frame_mode == "checkpoint"
        else args.temporal_ref_frame_mode
    )
    temporal_step = args.temporal_ref_frame_step or int(temporal_settings["ref_frame_step"])
    temporal_clip_frame_offsets = temporal_clip_offsets(
        int(temporal_settings["temporal_num_ref_frames"]),
        temporal_mode,
        temporal_step,
        str(temporal_settings["temporal_key_frame_position"]),
    )

    print(
        f"dataset={args.dataset_root} images={len(images)} annotations="
        f"{sum(len(coco.imgToAnns.get(image_id, [])) for image_id in image_ids)} "
        f"classes={id_to_name} device={device}",
        flush=True,
    )
    model_names = ["rf-detr-small", "temporal"] if args.models == "both" else [args.models]
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        if model_name == "rf-detr-small":
            wrapper = load_baseline(device)
            clip_offsets = [0]
            model_settings: dict[str, Any] = {"source": "pretrained COCO RF-DETR Small"}
        else:
            wrapper, model_settings = load_temporal(args.temporal_checkpoint, device)
            clip_offsets = temporal_clip_frame_offsets
        predictions = run_inference(
            model_name=model_name,
            wrapper=wrapper,
            images=images,
            image_dir=image_dir,
            device=device,
            shape=args.shape,
            score_threshold=args.coco_score_threshold,
            max_detections=args.max_detections,
            name_to_id=name_to_id,
            cache_path=args.output_dir / f"{model_name}_predictions.jsonl",
            groups=groups,
            positions=positions,
            clip_offsets=clip_offsets,
            overwrite=args.overwrite_predictions,
        )
        release_model(wrapper, device)

        predictions = [pred for pred in predictions if int(pred["image_id"]) in set(image_ids)]
        (args.output_dir / f"{model_name}_predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        fixed, fixed_per_class = fixed_threshold_metrics(
            coco, predictions, image_ids, args.operating_threshold, args.iou_threshold, id_to_name
        )
        coco_summary, coco_per_class, coco_text = run_coco_eval(
            coco, predictions, image_ids, category_ids, args.max_detections
        )
        (args.output_dir / f"{model_name}_coco_summary.txt").write_text(coco_text, encoding="utf-8")
        summary_rows.append(
            {
                "model": model_name,
                "images": len(images),
                "GT objects": fixed["gt"],
                f"predictions@{args.operating_threshold:g}": fixed["predictions"],
                "TP": fixed["tp"],
                "FP": fixed["fp"],
                "FN": fixed["fn"],
                "precision": fixed["precision"],
                "recall": fixed["recall"],
                "F1": fixed["F1"],
                **coco_summary,
            }
        )
        for row in fixed_per_class:
            category_id = int(row["category_id"])
            per_class_rows.append(
                {
                    "model": model_name,
                    **row,
                    **coco_per_class[category_id],
                }
            )
        (args.output_dir / f"{model_name}_model_settings.json").write_text(
            json.dumps(model_settings, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )

    write_csv(args.output_dir / "validation_summary.csv", summary_rows)
    write_csv(args.output_dir / "validation_per_class.csv", per_class_rows)
    payload = {
        "dataset_root": str(args.dataset_root),
        "annotation_path": str(annotation_path),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "device": str(device),
        "shape": args.shape,
        "coco_score_threshold": args.coco_score_threshold,
        "operating_threshold": args.operating_threshold,
        "iou_threshold": args.iou_threshold,
        "max_detections": args.max_detections,
        "classes": args.classes,
        "temporal_ref_frame_mode": temporal_mode,
        "temporal_ref_frame_step": temporal_step,
        "temporal_clip_offsets": temporal_clip_frame_offsets,
        "summary": summary_rows,
        "per_class": per_class_rows,
    }
    (args.output_dir / "validation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config_rows = [(key, value) for key, value in payload.items() if key not in {"summary", "per_class"}]
    write_excel(args.output_dir / "rfdetr_small_vs_temporal_validation.xlsx", summary_rows, per_class_rows, config_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
