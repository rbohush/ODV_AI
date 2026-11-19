#!/usr/bin/env python3
"""Run pretrained and temporal RF-DETR on one temporal COCO frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external/rf-detr/src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from validate_rfdetr_comparison import (  # noqa: E402
    TensorCache,
    build_video_index,
    load_baseline,
    load_temporal,
    select_clip_images,
    temporal_clip_offsets,
)


CLASS_NAMES = {1: "car", 2: "bus", 3: "truck"}
COLORS = {1: (255, 128, 0), 2: (0, 190, 255), 3: (0, 80, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image", required=True, help="COCO file_name to evaluate")
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--shape", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def find_frame(dataset_root: Path, file_name: str):
    for split in ("val", "train"):
        data = json.loads((dataset_root / "annotations" / f"instances_{split}2017.json").read_text())
        image = next((item for item in data["images"] if item["file_name"] == file_name), None)
        if image is not None:
            annotations = [item for item in data["annotations"] if int(item["image_id"]) == int(image["id"])]
            return split, data, image, annotations
    raise FileNotFoundError(f"{file_name} was not found in train or val annotations")


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def infer(wrapper, tensors, image, model_name: str, threshold: float):
    batch = torch.cat(tensors, dim=0).unsqueeze(0)
    with torch.inference_mode():
        outputs = wrapper.model.model(batch)
        sizes = torch.tensor([[image["height"], image["width"]]], device=batch.device)
        result = wrapper.model.postprocess(outputs, target_sizes=sizes)[0]
    mapping = {3: 1, 6: 2, 8: 3} if model_name == "rf-detr-small" else {1: 1, 2: 2, 3: 3}
    predictions = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        score_value = float(score.detach().cpu())
        category_id = mapping.get(int(label.detach().cpu()))
        if score_value >= threshold and category_id is not None:
            predictions.append({
                "category_id": category_id,
                "class_name": CLASS_NAMES[category_id],
                "score": score_value,
                "bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()],
            })
    return predictions


def evaluate(predictions, annotations, iou_threshold: float):
    gt = []
    for ann in annotations:
        x, y, width, height = map(float, ann["bbox"])
        gt.append({"category_id": int(ann["category_id"]), "bbox_xyxy": [x, y, x + width, y + height]})
    used = set()
    matches = []
    for gt_index, target in enumerate(gt):
        candidates = [
            (iou(target["bbox_xyxy"], prediction["bbox_xyxy"]), pred_index)
            for pred_index, prediction in enumerate(predictions)
            if pred_index not in used and prediction["category_id"] == target["category_id"]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        matched = best_index >= 0 and best_iou >= iou_threshold
        if matched:
            used.add(best_index)
        matches.append({"gt_index": gt_index, "matched": matched, "iou": best_iou, "prediction_index": best_index})
    true_positives = sum(item["matched"] for item in matches)
    return {
        "gt": len(gt),
        "predictions": len(predictions),
        "tp": true_positives,
        "fp": len(predictions) - true_positives,
        "fn": len(gt) - true_positives,
        "matches": matches,
    }


def draw(source: Path, destination: Path, model_name: str, image_meta, predictions, evaluation, threshold: float):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {source}")
    height, width = image.shape[:2]
    header_height = 42
    cv2.rectangle(image, (0, 0), (width, header_height), (18, 18, 18), -1)
    title = model_name
    font_scale = 0.62
    cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    thickness = max(2, round(min(height, width) / 360))
    label_scale = max(0.5, min(0.8, min(height, width) / 900))
    for prediction in predictions:
        x1, y1, x2, y2 = [int(round(value)) for value in prediction["bbox_xyxy"]]
        x1, y1 = max(0, x1), max(header_height, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        color = COLORS[prediction["category_id"]]
        label = f"{prediction['class_name']} {prediction['score'] * 100:.1f}%"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, thickness)
        top = max(header_height, y1 - text_height - base - 8)
        cv2.rectangle(image, (x1, top), (min(width - 1, x1 + text_width + 10), y1), color, -1)
        cv2.putText(image, label, (x1 + 5, max(top + text_height + 2, y1 - base - 3)), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(str(destination), image)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    split, data, image_meta, annotations = find_frame(args.dataset_root, args.image)
    image_dir = args.dataset_root / f"{split}2017"
    source = image_dir / args.image
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = TensorCache(image_dir, args.shape, device)

    baseline = load_baseline(device)
    baseline_predictions = infer(baseline, [cache.get(args.image)], image_meta, "rf-detr-small", args.threshold)
    baseline_eval = evaluate(baseline_predictions, annotations, args.iou_threshold)
    baseline_path = args.output_dir / f"rf-detr-small_{args.image}"
    draw(source, baseline_path, "rf-detr-small", image_meta, baseline_predictions, baseline_eval, args.threshold)
    del baseline
    if device.type == "mps":
        torch.mps.empty_cache()

    temporal, settings = load_temporal(args.temporal_checkpoint, device)
    offsets = temporal_clip_offsets(
        settings["temporal_num_ref_frames"], settings["ref_frame_mode"], settings["ref_frame_step"], settings["temporal_key_frame_position"]
    )
    groups, positions = build_video_index(data["images"])
    clip_images = select_clip_images(image_meta, groups, positions, offsets)
    temporal_predictions = infer(temporal, [cache.get(item["file_name"]) for item in clip_images], image_meta, "temporal", args.threshold)
    temporal_eval = evaluate(temporal_predictions, annotations, args.iou_threshold)
    temporal_path = args.output_dir / f"temporal_{args.image}"
    draw(source, temporal_path, "temporal", image_meta, temporal_predictions, temporal_eval, args.threshold)

    payload = {
        "image": image_meta,
        "split": split,
        "ground_truth": annotations,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "device": str(device),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "temporal_clip_offsets": offsets,
        "temporal_clip_files": [item["file_name"] for item in clip_images],
        "models": {
            "rf-detr-small": {"evaluation": baseline_eval, "predictions": baseline_predictions, "image": str(baseline_path)},
            "temporal": {"evaluation": temporal_eval, "predictions": temporal_predictions, "image": str(temporal_path)},
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
