#!/usr/bin/env python3
"""Extract first video halves and annotate every frame with both RF-DETR models.

Existing output files are kept, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = ROOT / "external" / "rf-detr" / "src"
for path in (RFDETR_SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_rfdetr_comparison import (  # noqa: E402
    load_baseline,
    load_temporal,
    temporal_clip_offsets,
)

BASELINE = {3: "car", 6: "bus", 8: "truck"}
TEMPORAL = {1: "car", 2: "bus", 3: "truck"}
COLORS = {"car": (255, 128, 0), "bus": (0, 190, 255), "truck": (0, 80, 255)}
HEADER = 42


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "videos" / "first_half_inference")
    parser.add_argument(
        "--temporal-checkpoint", type=Path,
        default=ROOT / "result" / "rfdetr_temporal_3cls_lora_frozen_rfdetr_noema_30ep" / "checkpoint_29.ckpt",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--shape", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def device_from(name: str) -> torch.device:
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def extract(video: Path, target_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = (total + 1) // 2
    if total <= 0:
        raise RuntimeError(f"Frame count unavailable for {video}")
    target_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index in range(limit):
        ok, image = cap.read()
        if not ok:
            break
        path = target_dir / f"{video.stem}_{index:06d}.jpg"
        if not path.exists() and not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {path}")
        frames.append(path)
    cap.release()
    print(f"{video.name}: extracted {len(frames)}/{total} frames")
    return frames


def tensor(path: Path, shape: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        value = F.to_tensor(image.convert("RGB"))
    value = F.resize(value, [shape, shape], antialias=True)
    return F.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)


def predictions(wrapper, tensors: list[torch.Tensor], size: tuple[int, int], mapping: dict[int, str], threshold: float) -> list[dict]:
    height, width = size
    batch = torch.cat(tensors, dim=0).unsqueeze(0)
    with torch.inference_mode():
        out = wrapper.model.model(batch)
        result = wrapper.model.postprocess(out, target_sizes=torch.tensor([[height, width]], device=batch.device))[0]
    result_list = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        score = float(score.detach().cpu())
        label = mapping.get(int(label.detach().cpu()))
        if label is not None and score >= threshold:
            result_list.append({"class": label, "score": score, "box": box.detach().cpu().tolist()})
    return result_list


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def draw(source: Path, target: Path, model_name: str, items: list[dict]) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {source}")
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, HEADER), (18, 18, 18), -1)
    cv2.putText(image, model_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    thickness = max(2, round(min(height, width) / 360))
    scale = max(0.5, min(0.8, min(height, width) / 900))
    labels: list[tuple[int, int, int, int]] = []
    for item in sorted(items, key=lambda value: value["score"], reverse=True):
        x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
        x1, y1, x2, y2 = max(0, x1), max(HEADER, y1), min(width - 1, x2), min(height - 1, y2)
        text = f"{item['class']} {item['score'] * 100:.1f}%"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        lw, lh = tw + 10, th + base + 8
        candidates = [(x1, y1 - lh, x1 + lw, y1), (x1, y2 + 2, x1 + lw, y2 + 2 + lh), (x2 + 2, y1, x2 + 2 + lw, y1 + lh), (x1 - lw - 2, y1, x1 - 2, y1 + lh)]
        valid = [rect for rect in candidates if rect[0] >= 0 and rect[1] >= HEADER and rect[2] < width and rect[3] < height and not any(overlaps(rect, old) for old in labels)]
        lx1, ly1, lx2, ly2 = valid[0] if valid else candidates[0]
        color = COLORS[item["class"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(image, text, (lx1 + 5, ly2 - base - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        labels.append((lx1, ly1, lx2, ly2))
    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    return image.shape[:2]


def run(model_name: str, wrapper, sequences: dict[str, list[Path]], output: Path, shape: int, device: torch.device, threshold: float, mapping: dict[int, str], offsets: list[int], overwrite: bool) -> dict[str, int]:
    rendered = skipped = 0
    for scene, frames in sequences.items():
        scene_output = output / model_name / scene
        scene_output.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(frames):
            target = scene_output / source.name
            if target.exists() and not overwrite:
                skipped += 1
                continue
            refs = [frames[min(max(index + offset, 0), len(frames) - 1)] for offset in offsets]
            output_items = predictions(wrapper, [tensor(path, shape, device) for path in refs], image_size(source), mapping, threshold)
            draw(source, target, model_name, output_items)
            rendered += 1
            if rendered % 25 == 0:
                print(f"{model_name}: rendered {rendered} frames")
    return {"rendered": rendered, "skipped": skipped}


def main() -> None:
    options = args()
    device = device_from(options.device)
    videos = sorted(options.videos_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos in {options.videos_dir}")
    raw = options.output_dir / "source_frames"
    sequences = {video.stem: extract(video, raw / video.stem) for video in videos}
    options.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(device)
    base_report = run("rf-detr-small", baseline, sequences, options.output_dir, options.shape, device, options.threshold, BASELINE, [0], options.overwrite)
    del baseline
    if device.type == "mps":
        torch.mps.empty_cache()
    temporal, config = load_temporal(options.temporal_checkpoint, device)
    offsets = temporal_clip_offsets(config["temporal_num_ref_frames"], config["ref_frame_mode"], config["ref_frame_step"], config["temporal_key_frame_position"])
    temporal_report = run("rf-detr-temporal", temporal, sequences, options.output_dir, config["resolution"], device, options.threshold, TEMPORAL, offsets, options.overwrite)
    report = {"device": str(device), "threshold": options.threshold, "checkpoint": str(options.temporal_checkpoint), "temporal_offsets": offsets, "source_frames": {scene: len(frames) for scene, frames in sequences.items()}, "rf-detr-small": base_report, "rf-detr-temporal": temporal_report}
    (options.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
