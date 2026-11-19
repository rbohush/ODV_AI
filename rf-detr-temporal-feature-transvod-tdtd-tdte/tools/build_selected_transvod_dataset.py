#!/usr/bin/env python3
"""Build the minimal scene subset needed for selected TransVOD++ inference."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SCENES = {
    "train": {"cars26", "test_cars18"},
    "val": {"test_cars12"},
}
REQUIRED_KEYS = {
    ("train", "cars26", 44),
    ("train", "test_cars18", 161),
    ("val", "test_cars12", 827),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use hardlink only when source and output are on the same filesystem.",
    )
    return parser.parse_args()


def filtered_coco(data: dict[str, Any], scenes: set[str]) -> dict[str, Any]:
    images = [
        image for image in data.get("images", []) if str(image.get("video_id")) in scenes
    ]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in data.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    output = {
        key: value
        for key, value in data.items()
        if key not in {"images", "annotations"}
    }
    output["images"] = images
    output["annotations"] = annotations
    return output


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Output already exists: {args.output_root}. Choose a new directory."
        )
    (args.output_root / "annotations").mkdir(parents=True)
    summary: dict[str, Any] = {"source_root": str(args.source_root.resolve()), "splits": {}}
    found_keys: set[tuple[str, str, int]] = set()

    for split, scenes in DEFAULT_SCENES.items():
        annotation_path = (
            args.source_root / "annotations" / f"instances_{split}2017.json"
        )
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        subset = filtered_coco(data, scenes)
        output_images = args.output_root / f"{split}2017"
        output_images.mkdir()
        scene_counts = Counter(str(image.get("video_id")) for image in subset["images"])
        annotation_counts = Counter()
        image_to_scene = {
            int(image["id"]): str(image.get("video_id")) for image in subset["images"]
        }
        for annotation in subset["annotations"]:
            annotation_counts[image_to_scene[int(annotation["image_id"])]] += 1

        copied_bytes = 0
        for image in subset["images"]:
            source = args.source_root / f"{split}2017" / str(image["file_name"])
            destination = output_images / str(image["file_name"])
            if not source.is_file():
                raise FileNotFoundError(source)
            transfer(source, destination, args.copy_mode)
            copied_bytes += source.stat().st_size
            key = (split, str(image.get("video_id")), int(image.get("frame_id", -1)))
            if key in REQUIRED_KEYS:
                found_keys.add(key)

        output_annotation = (
            args.output_root / "annotations" / f"instances_{split}2017.json"
        )
        output_annotation.write_text(
            json.dumps(subset, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        summary["splits"][split] = {
            "images": len(subset["images"]),
            "annotations": len(subset["annotations"]),
            "size_mb": copied_bytes / 1024**2,
            "images_by_scene": dict(sorted(scene_counts.items())),
            "annotations_by_scene": dict(sorted(annotation_counts.items())),
        }

    missing_keys = sorted(REQUIRED_KEYS - found_keys)
    if missing_keys:
        raise RuntimeError(f"Required key frames were not copied: {missing_keys}")
    summary["required_key_frames"] = [list(item) for item in sorted(found_keys)]
    summary_path = args.output_root / "subset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
