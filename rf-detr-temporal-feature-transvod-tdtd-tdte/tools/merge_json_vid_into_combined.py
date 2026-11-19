#!/usr/bin/env python3
"""Split JSON+VID scenes and merge them into temporal_dataset_combined."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


CANONICAL_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": ""},
    {"id": 2, "name": "bus", "supercategory": "vehicle"},
    {"id": 3, "name": "truck", "supercategory": "vehicle"},
]
CANONICAL_ID = {item["name"]: item["id"] for item in CANONICAL_CATEGORIES}
FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write JSON files and link images. Without this flag, only validate and report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def frame_number(file_name: str) -> int:
    match = FRAME_RE.search(Path(file_name).name)
    if not match:
        raise ValueError(f"Cannot extract frame number from {file_name!r}")
    return int(match.group(1))


def atomic_json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def category_counts(dataset: dict) -> Counter:
    names = {item["id"]: item["name"] for item in dataset["categories"]}
    return Counter(names[item["category_id"]] for item in dataset["annotations"])


def summarize(dataset: dict) -> dict:
    counts = category_counts(dataset)
    return {
        "images": len(dataset["images"]),
        "annotations": len(dataset["annotations"]),
        "car": counts["car"],
        "bus": counts["bus"],
        "truck": counts["truck"],
    }


def validate_dataset(dataset: dict, label: str) -> None:
    image_ids = [item["id"] for item in dataset["images"]]
    ann_ids = [item["id"] for item in dataset["annotations"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{label}: duplicate image IDs")
    if len(ann_ids) != len(set(ann_ids)):
        raise ValueError(f"{label}: duplicate annotation IDs")
    valid_images = set(image_ids)
    if any(item["image_id"] not in valid_images for item in dataset["annotations"]):
        raise ValueError(f"{label}: annotation references an unknown image")
    valid_categories = {item["id"] for item in CANONICAL_CATEGORIES}
    if any(item["category_id"] not in valid_categories for item in dataset["annotations"]):
        raise ValueError(f"{label}: unknown category ID")


def discover_scenes(source_root: Path) -> list[dict]:
    scenes = []
    scene_dirs = [path for path in source_root.glob("cars*") if path.is_dir()]
    scene_dirs.sort(key=lambda path: frame_number(path.name + ".jpg"))
    for outer in scene_dirs:
        scene = outer.name
        inner = outer / scene
        annotation_path = inner / "annotations.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        raw = load_json(annotation_path)
        source_categories = {item["id"]: item["name"] for item in raw["categories"]}
        unknown = set(source_categories.values()) - set(CANONICAL_ID)
        if unknown:
            raise ValueError(f"{scene}: unsupported categories: {sorted(unknown)}")

        images = []
        for image in raw["images"]:
            source_path = inner / Path(image["file_name"]).name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            item = copy.deepcopy(image)
            item["_source_path"] = source_path
            item["_frame_id"] = frame_number(image["file_name"])
            images.append(item)
        images.sort(key=lambda item: item["_frame_id"])
        if len({item["id"] for item in images}) != len(images):
            raise ValueError(f"{scene}: duplicate source image IDs")
        if len({item["_frame_id"] for item in images}) != len(images):
            raise ValueError(f"{scene}: duplicate frame numbers")

        annotations_by_image: dict[int, list[dict]] = {item["id"]: [] for item in images}
        for annotation in raw["annotations"]:
            if annotation["image_id"] not in annotations_by_image:
                raise ValueError(f"{scene}: annotation references missing image")
            annotations_by_image[annotation["image_id"]].append(annotation)
        scenes.append(
            {
                "name": scene,
                "images": images,
                "annotations_by_image": annotations_by_image,
                "source_categories": source_categories,
            }
        )
    if not scenes:
        raise ValueError(f"No cars* scenes found under {source_root}")
    return scenes


def append_scene_split(
    target: dict,
    scene: dict,
    images: list[dict],
    image_dir: Path,
    links: list[tuple[Path, Path]],
) -> dict:
    next_image_id = max((item["id"] for item in target["images"]), default=0) + 1
    next_ann_id = max((item["id"] for item in target["annotations"]), default=0) + 1
    source_to_target = {}
    counts = Counter()

    for source_image in images:
        frame_id = source_image["_frame_id"]
        file_name = f"{scene['name']}_{frame_id:06d}.jpg"
        if any(item["file_name"] == file_name for item in target["images"]):
            raise ValueError(f"Duplicate target file name: {file_name}")
        new_image = {
            "id": next_image_id,
            "file_name": file_name,
            "width": source_image["width"],
            "height": source_image["height"],
            "video_id": scene["name"],
            "frame_id": frame_id,
            "source_dataset": "JSON+VID",
            "source_image_id": source_image["id"],
            "source_split": "default",
        }
        target["images"].append(new_image)
        source_to_target[source_image["id"]] = next_image_id
        links.append((source_image["_source_path"], image_dir / file_name))
        next_image_id += 1

    for source_image in images:
        for source_annotation in scene["annotations_by_image"][source_image["id"]]:
            category_name = scene["source_categories"][source_annotation["category_id"]]
            new_annotation = copy.deepcopy(source_annotation)
            new_annotation["id"] = next_ann_id
            new_annotation["image_id"] = source_to_target[source_image["id"]]
            new_annotation["category_id"] = CANONICAL_ID[category_name]
            target["annotations"].append(new_annotation)
            counts[category_name] += 1
            next_ann_id += 1
    return {
        "images": len(images),
        "annotations": sum(counts.values()),
        "car": counts["car"],
        "bus": counts["bus"],
        "truck": counts["truck"],
    }


def link_images(links: list[tuple[Path, Path]]) -> None:
    for source, destination in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def markdown_report(report: dict) -> str:
    lines = [
        "# JSON+VID dataset merge report",
        "",
        f"Split rule: first 80% train, last {report['val_fraction']:.0%} validation per scene.",
        "",
        "| Scene | Train images | Val images | Train annotations | Val annotations | Car | Bus | Truck |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["scenes"]:
        lines.append(
            f"| {row['scene']} | {row['train']['images']} | {row['val']['images']} | "
            f"{row['train']['annotations']} | {row['val']['annotations']} | "
            f"{row['total']['car']} | {row['total']['bus']} | {row['total']['truck']} |"
        )
    lines.extend(["", "## Overall", "", "| Stage | Split | Images | Annotations | Car | Bus | Truck |", "|---|---|---:|---:|---:|---:|---:|"])
    for stage in ("before", "added", "after"):
        for split in ("train", "val", "total"):
            row = report[stage][split]
            lines.append(
                f"| {stage} | {split} | {row['images']} | {row['annotations']} | "
                f"{row['car']} | {row['bus']} | {row['truck']} |"
            )
    lines.extend(
        [
            "",
            "Old combined-dataset images were not present at merge time and are expected to be added manually.",
            "All newly added JSON+VID images are linked into train2017/val2017.",
            "",
        ]
    )
    return "\n".join(lines)


def add_rows(left: dict, right: dict) -> dict:
    return {key: left[key] + right[key] for key in left}


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    annotation_dir = args.target_root / "annotations"
    paths = {
        "train": annotation_dir / "instances_train2017.json",
        "val": annotation_dir / "instances_val2017.json",
    }
    original = {split: load_json(path) for split, path in paths.items()}
    for split, dataset in original.items():
        validate_dataset(dataset, f"original {split}")
    existing_scenes = {
        item.get("video_id") for dataset in original.values() for item in dataset["images"]
    }
    scenes = discover_scenes(args.source_root)
    overlap = sorted(scene["name"] for scene in scenes if scene["name"] in existing_scenes)
    if overlap:
        raise ValueError(f"Scenes already exist in target dataset: {overlap}")

    merged = {split: copy.deepcopy(dataset) for split, dataset in original.items()}
    for dataset in merged.values():
        dataset["categories"] = copy.deepcopy(CANONICAL_CATEGORIES)
        sources = dataset.setdefault("info", {}).setdefault("sources", [])
        if "JSON+VID" not in sources:
            sources.append("JSON+VID")

    links: list[tuple[Path, Path]] = []
    scene_rows = []
    added = {split: {key: 0 for key in summarize(original[split])} for split in ("train", "val")}
    for scene in scenes:
        val_count = max(1, math.ceil(len(scene["images"]) * args.val_fraction))
        train_images = scene["images"][:-val_count]
        val_images = scene["images"][-val_count:]
        train_stats = append_scene_split(
            merged["train"], scene, train_images, args.target_root / "train2017", links
        )
        val_stats = append_scene_split(
            merged["val"], scene, val_images, args.target_root / "val2017", links
        )
        added["train"] = add_rows(added["train"], train_stats)
        added["val"] = add_rows(added["val"], val_stats)
        scene_rows.append(
            {
                "scene": scene["name"],
                "train": train_stats,
                "val": val_stats,
                "total": add_rows(train_stats, val_stats),
                "train_frame_range": [train_images[0]["_frame_id"], train_images[-1]["_frame_id"]],
                "val_frame_range": [val_images[0]["_frame_id"], val_images[-1]["_frame_id"]],
            }
        )

    for split, dataset in merged.items():
        validate_dataset(dataset, f"merged {split}")
    before = {split: summarize(original[split]) for split in ("train", "val")}
    after = {split: summarize(merged[split]) for split in ("train", "val")}
    for group in (before, added, after):
        group["total"] = add_rows(group["train"], group["val"])
    report = {
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "val_fraction": args.val_fraction,
        "scenes": scene_rows,
        "before": before,
        "added": added,
        "after": after,
    }

    print(markdown_report(report))
    if not args.execute:
        print("Dry run complete. Add --execute to write the merged dataset.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = annotation_dir / f"backup_before_JSON_VID_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for split, path in paths.items():
        shutil.copy2(path, backup_dir / path.name)
    link_images(links)
    for split, path in paths.items():
        atomic_json_dump(path, merged[split])
    report_dir = args.target_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report_dir / "json_vid_merge_summary.json", report)
    (report_dir / "json_vid_merge_summary.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(f"Merge complete. Backup: {backup_dir}")


if __name__ == "__main__":
    main()
