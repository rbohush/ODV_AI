#!/usr/bin/env python3
"""Evaluate temporal detection stability metrics on a COCO video scene."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

DATASET_CAT_NAMES = {1: "car", 2: "bus", 3: "truck"}
COCO_SLOT_TO_DATASET_CAT = {3: 1, 6: 2, 8: 3}
LOCAL_SLOT_TO_DATASET_CAT = {1: 1, 2: 2, 3: 3}
COLORS = {1: (255, 128, 0), 2: (0, 180, 255), 3: (0, 80, 255)}


@dataclass(frozen=True)
class FrameInfo:
    split: str
    image: dict[str, Any]
    path: Path
    annotations: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("temporal_dataset_new"))
    parser.add_argument("--scene", default="test_cars18")
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/temporal_metrics_test_cars18"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--shape", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--temporal-ref-frame-mode", choices=["previous", "surrounding", "duplicate", "checkpoint"], default="checkpoint")
    parser.add_argument("--temporal-ref-frame-step", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--draw-frames", action="store_true")
    parser.add_argument("--make-videos", action="store_true")
    parser.add_argument(
        "--temporal-model-name",
        default="temporal_9_epoch",
        help="Display/output name for the temporal checkpoint in reports and annotated frame headers.",
    )
    return parser.parse_args()


def normalize_track_id(annotation: dict[str, Any]) -> str:
    value = annotation.get("track_id", (annotation.get("attributes") or {}).get("track_id"))
    return str(value) if value is not None else f"ann_{annotation['id']}"


def load_scene(dataset_root: Path, scene: str, max_frames: int) -> list[FrameInfo]:
    frames: list[FrameInfo] = []
    for split in ("val", "train"):
        ann_path = dataset_root / "annotations" / f"instances_{split}2017.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)
        image_dir = dataset_root / f"{split}2017"
        for image in data.get("images", []):
            if image.get("video_id") != scene:
                continue
            frames.append(
                FrameInfo(
                    split=split,
                    image=image,
                    path=image_dir / image["file_name"],
                    annotations=tuple(anns_by_image.get(int(image["id"]), [])),
                )
            )
    frames.sort(key=lambda item: (int(item.image.get("frame_id", 0)), int(item.image["id"])))
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"Scene {scene!r} was not found in {dataset_root}")
    missing = [frame.path for frame in frames if not frame.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image files: {missing[:5]}")
    return frames


def temporal_offsets(num_refs: int, mode: str, step: int) -> list[int]:
    if num_refs <= 0:
        return []
    if mode == "duplicate":
        return [0] * num_refs
    if mode == "previous":
        return [-(idx + 1) * step for idx in range(num_refs)]
    if mode == "surrounding":
        offsets: list[int] = []
        radius = 1
        while len(offsets) < num_refs:
            offsets.extend([-radius * step, radius * step])
            radius += 1
        return offsets[:num_refs]
    raise ValueError(f"Unknown mode: {mode}")


def preprocess(path: Path, shape: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = F.to_tensor(image).to(device)
    tensor = F.resize(tensor, [shape, shape], antialias=True)
    return F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def iou_xyxy(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y + h]


def checkpoint_train_config(wrapper: Any) -> dict[str, Any]:
    args = getattr(getattr(wrapper, "model", None), "args", None)
    train_config = getattr(args, "train_config", None)
    return train_config if isinstance(train_config, dict) else {}


def load_baseline(device: torch.device) -> Any:
    from rfdetr import RFDETRSmall
    wrapper = RFDETRSmall()
    wrapper.model.device = device
    wrapper.model.model.to(device).eval()
    return wrapper


def load_temporal(path: Path, device: torch.device) -> Any:
    from rfdetr import RFDETRSmall, from_checkpoint
    if path.suffix == ".ckpt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = RFDETRSmall(
            num_classes=3,
            resolution=640,
            backbone_lora=True,
            temporal_num_ref_frames=3,
            temporal_fusion_layers=1,
            temporal_key_frame_position="center",
            temporal_tdte_layers=1,
            temporal_tdtd_layers=1,
        )
        state = {key.removeprefix("model."): value for key, value in ckpt["state_dict"].items()}
        incompatible = wrapper.model.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"temporal checkpoint loaded with missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )
    else:
        wrapper = from_checkpoint(path)
    wrapper.model.device = device
    wrapper.model.model.to(device).eval()
    return wrapper


def map_label(model_name: str, label: int) -> int | None:
    if model_name == "rf-detr-small":
        return COCO_SLOT_TO_DATASET_CAT.get(int(label))
    return LOCAL_SLOT_TO_DATASET_CAT.get(int(label))


def run_model(
    model_name: str,
    wrapper: Any,
    frames: list[FrameInfo],
    device: torch.device,
    shape: int,
    threshold: float,
    temporal_mode_arg: str,
    temporal_step_arg: int,
) -> list[dict[str, Any]]:
    num_refs = int(getattr(wrapper.model_config, "temporal_num_ref_frames", 0))
    train_cfg = checkpoint_train_config(wrapper)
    mode = temporal_mode_arg
    if mode == "checkpoint":
        mode = str(train_cfg.get("temporal_ref_frame_mode", "previous"))
    step = temporal_step_arg or int(train_cfg.get("temporal_ref_frame_step", 1))
    offsets = temporal_offsets(num_refs, mode, step) if num_refs > 0 else []
    predictions: list[dict[str, Any]] = []
    print(f"{model_name}: frames={len(frames)} refs={num_refs} mode={mode} step={step} threshold={threshold}", flush=True)
    with torch.inference_mode():
        for idx, frame in enumerate(frames):
            clip_indices = [idx]
            for offset in offsets:
                clip_indices.append(min(max(idx + offset, 0), len(frames) - 1))
            tensor = torch.cat([preprocess(frames[j].path, shape, device) for j in clip_indices], dim=0).unsqueeze(0)
            outputs = wrapper.model.model(tensor)
            target_sizes = torch.tensor([[frame.image["height"], frame.image["width"]]], device=device)
            result = wrapper.model.postprocess(outputs, target_sizes=target_sizes)[0]
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            for score, label, box in zip(scores, labels, boxes):
                if float(score) < threshold:
                    continue
                dataset_cat = map_label(model_name, int(label))
                if dataset_cat is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                predictions.append(
                    {
                        "frame_index": idx,
                        "frame_id": int(frame.image.get("frame_id", idx)),
                        "image_id": int(frame.image["id"]),
                        "category_id": dataset_cat,
                        "class_name": DATASET_CAT_NAMES[dataset_cat],
                        "score": float(score),
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
            if device.type == "mps" and (idx + 1) % 25 == 0:
                torch.mps.empty_cache()
            if (idx + 1) % 50 == 0:
                print(f"{model_name}: processed {idx + 1}/{len(frames)}", flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()
    return predictions


def match_predictions(frames: list[FrameInfo], predictions: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    preds_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        preds_by_frame[int(pred["frame_id"])].append(pred)
    track_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_records: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = int(frame.image.get("frame_id", 0))
        preds = preds_by_frame.get(frame_id, [])
        used: set[int] = set()
        gt_items = []
        for ann in frame.annotations:
            cat = int(ann["category_id"])
            track_id = normalize_track_id(ann)
            gt_box = xywh_to_xyxy(ann["bbox"])
            gt_items.append((track_id, cat, gt_box, ann))
        for track_id, cat, gt_box, ann in gt_items:
            best_idx = None
            best_iou = 0.0
            best_score = 0.0
            best_box = None
            for pred_idx, pred in enumerate(preds):
                if pred_idx in used or int(pred["category_id"]) != cat:
                    continue
                score = float(pred["score"])
                pred_iou = iou_xyxy(gt_box, pred["bbox_xyxy"])
                if pred_iou > best_iou or (math.isclose(pred_iou, best_iou) and score > best_score):
                    best_idx = pred_idx
                    best_iou = pred_iou
                    best_score = score
                    best_box = pred["bbox_xyxy"]
            detected = best_idx is not None and best_iou >= iou_threshold
            if detected and best_idx is not None:
                used.add(best_idx)
            record = {
                "frame_id": frame_id,
                "track_id": track_id,
                "category_id": cat,
                "class_name": DATASET_CAT_NAMES.get(cat, str(cat)),
                "gt_bbox_xyxy": gt_box,
                "detected": bool(detected),
                "iou": float(best_iou if detected else 0.0),
                "score": float(best_score if detected else 0.0),
                "pred_bbox_xyxy": best_box if detected else None,
                "occluded": bool((ann.get("attributes") or {}).get("occluded", ann.get("occluded", False))),
            }
            track_records[track_id].append(record)
            frame_records.append(record)
    return {"tracks": dict(track_records), "frames": frame_records}


def contiguous_pairs(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(records, key=lambda r: int(r["frame_id"]))
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if int(b["frame_id"]) == int(a["frame_id"]) + 1]


def detected_segments(records: list[dict[str, Any]]) -> int:
    count = 0
    previous = False
    for record in sorted(records, key=lambda r: int(r["frame_id"])):
        current = bool(record["detected"])
        if current and not previous:
            count += 1
        previous = current
    return count


def recovery_events(records: list[dict[str, Any]], max_gap: int = 10, recovery_window: int = 3) -> list[bool]:
    ordered = sorted(records, key=lambda r: int(r["frame_id"]))
    events: list[bool] = []
    prev = ordered[0] if ordered else None
    for idx, record in enumerate(ordered[1:], start=1):
        if prev is None:
            prev = record
            continue
        gap = int(record["frame_id"]) - int(prev["frame_id"]) - 1
        if 1 <= gap <= max_gap:
            window = ordered[idx : min(len(ordered), idx + recovery_window)]
            events.append(any(bool(item["detected"]) for item in window))
        prev = record
    return events


def compute_metrics(matched: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_track: list[dict[str, Any]] = []
    all_records = matched["frames"]
    total_gt = len(all_records)
    total_detected = sum(bool(r["detected"]) for r in all_records)
    flickers = 0
    adjacent_presence = 0
    pred_stability_ious: list[float] = []
    gt_motion_ious: list[float] = []
    recovery: list[bool] = []
    occluded_records = [r for r in all_records if r["occluded"]]

    for track_id, records in matched["tracks"].items():
        records = sorted(records, key=lambda r: int(r["frame_id"]))
        detected = sum(bool(r["detected"]) for r in records)
        pairs = contiguous_pairs(records)
        track_flickers = sum(1 for a, b in pairs if bool(a["detected"]) != bool(b["detected"]))
        stable_pairs = [
            iou_xyxy(a["pred_bbox_xyxy"], b["pred_bbox_xyxy"])
            for a, b in pairs
            if a["detected"] and b["detected"] and a["pred_bbox_xyxy"] is not None and b["pred_bbox_xyxy"] is not None
        ]
        gt_pair_ious = [iou_xyxy(a["gt_bbox_xyxy"], b["gt_bbox_xyxy"]) for a, b in pairs]
        track_recovery = recovery_events(records)
        recovery.extend(track_recovery)
        flickers += track_flickers
        adjacent_presence += len(pairs)
        pred_stability_ious.extend(stable_pairs)
        gt_motion_ious.extend(gt_pair_ious)
        per_track.append(
            {
                "track_id": track_id,
                "class_name": records[0]["class_name"],
                "frames": len(records),
                "detected_frames": detected,
                "missed_frames": len(records) - detected,
                "recall": detected / len(records) if records else float("nan"),
                "flicker_events": track_flickers,
                "flicker_rate": track_flickers / len(pairs) if pairs else float("nan"),
                "detected_segments": detected_segments(records),
                "continuity": max_consecutive_detected(records) / len(records) if records else float("nan"),
                "box_stability_iou": float(np.mean(stable_pairs)) if stable_pairs else float("nan"),
                "box_instability": 1.0 - float(np.mean(stable_pairs)) if stable_pairs else float("nan"),
                "recovery_events": len(track_recovery),
                "temporary_occlusion_recall": float(np.mean(track_recovery)) if track_recovery else float("nan"),
            }
        )
    summary = {
        "gt_objects": total_gt,
        "detected_gt_objects": total_detected,
        "missed_gt_objects": total_gt - total_detected,
        "recall": total_detected / total_gt if total_gt else float("nan"),
        "miss_rate": 1.0 - (total_detected / total_gt) if total_gt else float("nan"),
        "detection_flicker_events": flickers,
        "detection_flicker_rate": flickers / adjacent_presence if adjacent_presence else float("nan"),
        "track_continuity_mean": float(np.nanmean([t["continuity"] for t in per_track])) if per_track else float("nan"),
        "detected_segments_total": int(sum(t["detected_segments"] for t in per_track)),
        "box_stability_iou_mean": float(np.mean(pred_stability_ious)) if pred_stability_ious else float("nan"),
        "box_instability_mean": 1.0 - float(np.mean(pred_stability_ious)) if pred_stability_ious else float("nan"),
        "gt_frame_to_frame_iou_mean": float(np.mean(gt_motion_ious)) if gt_motion_ious else float("nan"),
        "temporary_occlusion_events": len(recovery),
        "temporary_occlusion_recall": float(np.mean(recovery)) if recovery else None,
        "annotated_occluded_gt_objects": len(occluded_records),
        "annotated_occlusion_recall": (sum(r["detected"] for r in occluded_records) / len(occluded_records)) if occluded_records else None,
    }
    return summary, sorted(per_track, key=lambda r: (r["class_name"], int(r["track_id"]) if str(r["track_id"]).isdigit() else str(r["track_id"])))


def max_consecutive_detected(records: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    prev_frame = None
    for record in sorted(records, key=lambda r: int(r["frame_id"])):
        frame_id = int(record["frame_id"])
        if record["detected"] and (prev_frame is None or frame_id == prev_frame + 1):
            current += 1
        elif record["detected"]:
            current = 1
        else:
            current = 0
        best = max(best, current)
        prev_frame = frame_id
    return best


def draw_outputs(model_name: str, frames: list[FrameInfo], predictions: list[dict[str, Any]], matched: dict[str, Any], out_dir: Path, make_video: bool) -> None:
    frames_dir = out_dir / model_name / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    preds_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        preds_by_frame[int(pred["frame_id"])].append(pred)
    matched_by_frame_track = {(r["frame_id"], r["track_id"]): r for r in matched["frames"]}
    writer = None
    try:
        for idx, frame in enumerate(frames):
            image_bgr = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Could not read {frame.path}")
            h, w = image_bgr.shape[:2]
            if make_video and writer is None:
                video_path = out_dir / model_name / f"{model_name}_{frame.image.get('video_id')}_annotated.mp4"
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w - w % 2, h - h % 2))
            cv2.rectangle(image_bgr, (0, 0), (w, 40), (18, 18, 18), -1)
            cv2.putText(image_bgr, f"{model_name} | {frame.image.get('video_id')} | frame {int(frame.image.get('frame_id', idx)):06d} | score >= 0.30", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
            frame_id = int(frame.image.get("frame_id", idx))
            for ann in frame.annotations:
                cat = int(ann["category_id"])
                track_id = normalize_track_id(ann)
                match = matched_by_frame_track.get((frame_id, track_id))
                label = f"{DATASET_CAT_NAMES.get(cat, cat)}_{track_id}"
                if match and match["detected"]:
                    label += f" {match['score'] * 100:.1f}%"
                    color = COLORS.get(cat, (0, 255, 0))
                else:
                    label += " missed"
                    color = (70, 70, 255)
                x1, y1, x2, y2 = [int(round(v)) for v in xywh_to_xyxy(ann["bbox"])]
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                top = max(0, y1 - th - base - 8)
                cv2.rectangle(image_bgr, (x1, top), (min(w - 1, x1 + tw + 10), y1), color, -1)
                cv2.putText(image_bgr, label, (x1 + 5, max(th + 2, y1 - base - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            # Draw unmatched predictions with thin gray boxes for false positives / extra detections.
            matched_pred_boxes = [tuple(r["pred_bbox_xyxy"] or []) for r in matched["frames"] if r["frame_id"] == frame_id and r["detected"]]
            for pred in preds_by_frame.get(frame_id, []):
                if any(pred["bbox_xyxy"] and iou_xyxy(pred["bbox_xyxy"], box) > 0.999 for box in matched_pred_boxes if box):
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox_xyxy"]]
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (160, 160, 160), 1)
                cv2.putText(image_bgr, f"{pred['class_name']} {pred['score']*100:.1f}%", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            out_path = frames_dir / f"frame_id_{frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), image_bgr)
            if writer is not None:
                writer.write(image_bgr[: h - h % 2, : w - w % 2])
    finally:
        if writer is not None:
            writer.release()


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("recall", "Recall"),
        ("miss_rate", "Miss Rate"),
        ("detection_flicker_rate", "Detection Flicker Rate"),
        ("track_continuity_mean", "Track Continuity"),
        ("box_stability_iou_mean", "Frame-to-frame Box Stability"),
        ("box_instability_mean", "Box Instability"),
    ]
    names = [row["model"] for row in summary_rows]
    for key, title in metrics:
        values = [row.get(key) for row in summary_rows]
        if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, values, color=["#1f77b4", "#ff7f0e"][: len(names)])
        ax.set_ylabel(title)
        ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        ax.grid(axis="y", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close(fig)


def plot_track_recall(model_tracks: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    track_ids = sorted({row["track_id"] for rows in model_tracks.values() for row in rows}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    x = np.arange(len(track_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(track_ids) * 0.55), 4.5))
    for idx, (model, rows) in enumerate(model_tracks.items()):
        by_track = {row["track_id"]: row for row in rows}
        values = [by_track.get(track, {}).get("recall", 0.0) for track in track_ids]
        ax.bar(x + (idx - 0.5) * width, values, width, label=model)
    ax.set_ylabel("Recall")
    ax.set_xlabel("GT track_id")
    ax.set_xticks(x)
    ax.set_xticklabels(track_ids)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "per_track_recall.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    frames = load_scene(args.dataset_root, args.scene, args.max_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"scene={args.scene} frames={len(frames)} frame_range={frames[0].image.get('frame_id')}..{frames[-1].image.get('frame_id')}")

    models = {
        "rf-detr-small": load_baseline(device),
        args.temporal_model_name: load_temporal(args.temporal_checkpoint, device),
    }
    summary_rows: list[dict[str, Any]] = []
    model_tracks: dict[str, list[dict[str, Any]]] = {}
    all_payload: dict[str, Any] = {
        "scene": args.scene,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "device": str(device),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "frames": len(frames),
        "frame_id_range": [int(frames[0].image.get("frame_id", 0)), int(frames[-1].image.get("frame_id", 0))],
        "models": {},
    }

    for model_name, wrapper in models.items():
        predictions = run_model(model_name, wrapper, frames, device, args.shape, args.threshold, args.temporal_ref_frame_mode, args.temporal_ref_frame_step)
        matched = match_predictions(frames, predictions, args.iou_threshold)
        summary, per_track = compute_metrics(matched)
        row = {"model": model_name, **summary}
        summary_rows.append(row)
        model_tracks[model_name] = per_track
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (model_dir / "matched_gt.json").write_text(json.dumps(matched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (model_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_csv(model_dir / "per_track_metrics.csv", per_track)
        if args.draw_frames:
            draw_outputs(model_name, frames, predictions, matched, args.output_dir, args.make_videos)
        all_payload["models"][model_name] = {"summary": summary, "per_track": per_track, "predictions": len(predictions)}

    save_csv(args.output_dir / "temporal_metrics_summary.csv", summary_rows)
    (args.output_dir / "temporal_metrics_summary.json").write_text(json.dumps(all_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_summaries(summary_rows, args.output_dir / "plots")
    plot_track_recall(model_tracks, args.output_dir / "plots")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
