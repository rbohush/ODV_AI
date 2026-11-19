#!/usr/bin/env python3
"""Build one clean and one ground-truth video for every logical dataset scene."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


TRAILING_NUMBER = re.compile(r"(\d+)$")
DANDELION_NAME = re.compile(r"^dandelion_(?:train|val)_(.+)_(\d+)$")


@dataclass(frozen=True)
class FrameRecord:
    scene: str
    frame_number: int
    split: str
    image: dict[str, Any]
    annotations: tuple[dict[str, Any], ...]
    image_path: Path
    categories: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("temporal_dataset_new"))
    parser.add_argument("--output-dir", type=Path, default=Path("videos"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=0,
        help="Resize output when its longest edge exceeds this value. 0 preserves source resolution.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def logical_scene(image: dict[str, Any]) -> tuple[str, int]:
    file_name = Path(image["file_name"])
    stem = file_name.stem
    source_dataset = str(image.get("source_dataset", "")).lower()
    is_dandelion = stem.startswith("dandelion_") or "dandelion" in source_dataset

    if is_dandelion:
        match = DANDELION_NAME.match(stem)
        if match is None:
            raise ValueError(f"Cannot parse Dandelion frame number from {file_name}")
        source_scene, source_frame = match.group(1), int(match.group(2))
        if source_scene in {"train2017", "val2017"}:
            scene = f"dandelion_static_{int(image['width'])}x{int(image['height'])}"
        else:
            scene = f"dandelion_{source_scene}"
        return scene, source_frame

    frame_id = image.get("frame_id")
    if isinstance(frame_id, int):
        return str(image.get("video_id", stem)), frame_id

    number_match = TRAILING_NUMBER.search(stem)
    frame_number = int(number_match.group(1)) if number_match else int(image["id"])
    return str(image.get("video_id", stem)), frame_number


def load_records(dataset_root: Path) -> dict[str, list[FrameRecord]]:
    scenes: dict[str, list[FrameRecord]] = defaultdict(list)
    missing: list[Path] = []

    for split in ("train", "val"):
        ann_path = dataset_root / "annotations" / f"instances_{split}2017.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        image_dir = dataset_root / f"{split}2017"
        for image in data.get("images", []):
            scene, frame_number = logical_scene(image)
            path = image_dir / image["file_name"]
            if not path.is_file():
                missing.append(path)
                continue
            scenes[scene].append(
                FrameRecord(
                    scene=scene,
                    frame_number=frame_number,
                    split=split,
                    image=image,
                    annotations=tuple(annotations_by_image.get(int(image["id"]), [])),
                    image_path=path,
                    categories=categories,
                )
            )

    if missing:
        sample = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} dataset images are missing. First paths:\n{sample}")

    for scene, records in scenes.items():
        records.sort(key=lambda item: (item.frame_number, 0 if item.split == "train" else 1, int(item.image["id"])))
        duplicate_numbers = [number for number, count in Counter(item.frame_number for item in records).items() if count > 1]
        if duplicate_numbers:
            sample = duplicate_numbers[:10]
            raise ValueError(f"Scene {scene!r} has duplicate frame numbers: {sample}")

    return dict(sorted(scenes.items()))


def output_size(width: int, height: int, max_long_edge: int) -> tuple[int, int, float]:
    if max_long_edge <= 0 or max(width, height) <= max_long_edge:
        # Common MP4 codecs require even dimensions and otherwise crop silently.
        return width - width % 2, height - height % 2, 1.0
    scale = max_long_edge / max(width, height)
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    out_width -= out_width % 2
    out_height -= out_height % 2
    return out_width, out_height, scale


def draw_ground_truth(frame: Any, record: FrameRecord, scale: float) -> Any:
    annotated = frame.copy()
    thickness = max(2, int(round(min(annotated.shape[:2]) / 360)))
    font_scale = max(0.5, min(1.2, min(annotated.shape[:2]) / 900))
    label_counts: Counter[str] = Counter()

    for annotation in record.annotations:
        x, y, width, height = (float(value) * scale for value in annotation["bbox"])
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        class_name = record.categories.get(int(annotation["category_id"]), str(annotation["category_id"]))
        label_counts[class_name] += 1
        track_id = annotation.get("track_id")
        label = f"{class_name}_{track_id}" if track_id is not None else class_name

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), thickness)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, top),
            (min(annotated.shape[1] - 1, x1 + text_width + 8), y1),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    header = f"{record.scene} | frame {record.frame_number} | {record.split}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], max(30, int(42 * font_scale))), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        header,
        (10, max(22, int(30 * font_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def open_writer(path: Path, codec: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path} with codec {codec}")
    return writer


def encode_scene(
    scene: str,
    records: list[FrameRecord],
    clean_path: Path,
    annotated_path: Path,
    fps: float,
    codec: str,
    max_long_edge: int,
    overwrite: bool,
) -> dict[str, Any]:
    if clean_path.exists() and annotated_path.exists() and not overwrite:
        return {"scene": scene, "status": "skipped_existing", "frames": len(records)}

    first = cv2.imread(str(records[0].image_path), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode {records[0].image_path}")
    source_height, source_width = first.shape[:2]
    out_width, out_height, scale = output_size(source_width, source_height, max_long_edge)
    size = (out_width, out_height)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    clean_writer = open_writer(clean_path, codec, fps, size)
    annotated_writer = open_writer(annotated_path, codec, fps, size)

    split_counts: Counter[str] = Counter()
    annotation_count = 0
    try:
        for record in records:
            frame = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not decode {record.image_path}")
            height, width = frame.shape[:2]
            if (width, height) != (source_width, source_height):
                raise ValueError(
                    f"Scene {scene!r} changes resolution at {record.image_path}: "
                    f"{width}x{height} instead of {source_width}x{source_height}"
                )
            if size != (source_width, source_height):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

            clean_writer.write(frame)
            annotated_writer.write(draw_ground_truth(frame, record, scale))
            split_counts[record.split] += 1
            annotation_count += len(record.annotations)
    finally:
        clean_writer.release()
        annotated_writer.release()

    return {
        "scene": scene,
        "status": "written",
        "frames": len(records),
        "train_frames": split_counts["train"],
        "val_frames": split_counts["val"],
        "annotations": annotation_count,
        "first_frame_number": records[0].frame_number,
        "last_frame_number": records[-1].frame_number,
        "source_size": [source_width, source_height],
        "output_size": [out_width, out_height],
        "clean_video": str(clean_path),
        "annotated_video": str(annotated_path),
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    scenes = load_records(args.dataset_root)
    clean_dir = args.output_dir / "clean"
    annotated_dir = args.output_dir / "annotated"
    manifest: list[dict[str, Any]] = []

    print(f"Found {len(scenes)} logical scenes")
    for index, (scene, records) in enumerate(scenes.items(), start=1):
        print(f"[{index}/{len(scenes)}] {scene}: {len(records)} frames", flush=True)
        manifest.append(
            encode_scene(
                scene=scene,
                records=records,
                clean_path=clean_dir / f"{scene}.mp4",
                annotated_path=annotated_dir / f"{scene}.mp4",
                fps=args.fps,
                codec=args.codec,
                max_long_edge=args.max_long_edge,
                overwrite=args.overwrite,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "dataset_root": str(args.dataset_root.resolve()),
        "fps": args.fps,
        "codec": args.codec,
        "max_long_edge": args.max_long_edge,
        "scene_count": len(manifest),
        "total_frames": sum(item["frames"] for item in manifest),
        "scenes": manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved clean videos to {clean_dir}")
    print(f"Saved annotated videos to {annotated_dir}")
    print(f"Saved manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
