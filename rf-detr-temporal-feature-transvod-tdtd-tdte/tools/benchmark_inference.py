from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torchvision.transforms import functional as transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"

if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark trained RF-DETR and Temporal RF-DETR inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=["nano", "small", "medium", "large"], default="small")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--backbone-lora", action="store_true")
    parser.add_argument("--temporal-num-ref-frames", type=int, default=0)
    parser.add_argument("--temporal-fusion-layers", type=int, default=1)
    parser.add_argument("--temporal-key-frame-position", choices=["first", "center"], default="first")
    parser.add_argument("--temporal-tdte-layers", type=int, default=0)
    parser.add_argument("--temporal-tdtd-layers", type=int, default=0)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--benchmark-mode",
        choices=["pure", "e2e", "dataset", "both"],
        default="pure",
        help="Pure model forward, repeated-image e2e, real dataset, or both (default: pure).",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--classes", nargs="+", default=["car", "bus"])
    parser.add_argument("--label-space", choices=["local", "coco"], default="local")
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["previous", "surrounding", "duplicate"],
        default="surrounding",
    )
    parser.add_argument("--temporal-ref-frame-step", type=int, default=2)
    parser.add_argument(
        "--input-images",
        type=Path,
        nargs="+",
        help=(
            "Images for e2e mode, ordered key then references. Provide T+1 paths "
            "(reused across the batch) or batch_size*(T+1) paths."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.batch_size < 1 or args.warmup < 0 or args.iters < 1 or args.max_images < 1:
        raise ValueError("batch-size and iters must be positive; warmup must be non-negative")
    if args.resolution < 1 or args.temporal_num_ref_frames < 0:
        raise ValueError("resolution must be positive and temporal-num-ref-frames non-negative")
    if args.dtype == "fp16" and not str(args.device).startswith("cuda"):
        raise ValueError("fp16 benchmark is supported only on CUDA")

    if args.benchmark_mode == "dataset":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for dataset mode")
        annotation_path = args.dataset_root / "annotations" / f"instances_{args.split}2017.json"
        image_dir = args.dataset_root / f"{args.split}2017"
        if not annotation_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {annotation_path} and {image_dir} for dataset mode"
            )
        if args.batch_size != 1:
            raise ValueError("dataset mode currently requires --batch-size 1")

    if args.benchmark_mode in {"e2e", "both"}:
        if not args.input_images:
            raise ValueError("--input-images is required for e2e/both mode")
        frames_per_clip = args.temporal_num_ref_frames + 1
        valid_counts = {frames_per_clip, args.batch_size * frames_per_clip}
        if len(args.input_images) not in valid_counts:
            raise ValueError(
                f"Expected {frames_per_clip} or {args.batch_size * frames_per_clip} "
                f"input image paths, got {len(args.input_images)}"
            )
        missing = [str(path) for path in args.input_images if not path.is_file()]
        if missing:
            raise FileNotFoundError("Input images do not exist: " + ", ".join(missing))


def get_model_class(variant: str) -> type:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    return {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }[variant]


def load_wrapper(args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "num_classes": args.num_classes,
        "resolution": args.resolution,
        "backbone_lora": args.backbone_lora,
    }
    if args.temporal_num_ref_frames > 0:
        kwargs.update(
            temporal_num_ref_frames=args.temporal_num_ref_frames,
            temporal_fusion_layers=args.temporal_fusion_layers,
            temporal_key_frame_position=args.temporal_key_frame_position,
            temporal_tdte_layers=args.temporal_tdte_layers,
            temporal_tdtd_layers=args.temporal_tdtd_layers,
            temporal_dropout=args.temporal_dropout,
        )
    model_class = get_model_class(args.variant)
    if args.checkpoint.suffix != ".ckpt":
        try:
            return model_class.from_checkpoint(args.checkpoint, **kwargs)
        except ValueError as error:
            if "Could not infer model class from checkpoint" not in str(error):
                raise
            print(
                "Checkpoint has no RF-DETR training metadata; loading it as "
                f"{model_class.__name__} pretrained weights.",
                flush=True,
            )
            return model_class(pretrain_weights=str(args.checkpoint), **kwargs)

    wrapper = model_class(**kwargs)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("model."): value for key, value in state.items()}
    incompatible = wrapper.model.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Lightning checkpoint load: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    return wrapper


def get_core_model(wrapper: Any) -> torch.nn.Module:
    inference_model = getattr(wrapper.model, "inference_model", None)
    return inference_model if inference_model is not None else wrapper.model.model


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def latency_metrics(times_ms: list[float], batch_size: int) -> dict[str, float]:
    mean_ms = statistics.mean(times_ms)
    fps = 1000.0 / mean_ms
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": statistics.median(times_ms),
        "latency_p95_ms": percentile_nearest_rank(times_ms, 0.95),
        "fps": fps,
        "key_frames_per_second": batch_size * fps,
    }


def reset_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_warmup(forward: Callable[[], Any], count: int, device: torch.device) -> None:
    for _ in range(count):
        forward()
    synchronize(device)


def measure_cuda_forward(forward: Callable[[], Any], iters: int, device: torch.device) -> list[float]:
    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def measure_wall_clock(forward: Callable[[], Any], iters: int, device: torch.device) -> list[float]:
    times_ms: list[float] = []
    for _ in range(iters):
        synchronize(device)
        started = time.perf_counter()
        forward()
        synchronize(device)
        times_ms.append((time.perf_counter() - started) * 1000.0)
    return times_ms


def add_peak_memory(result: dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        result["peak_vram_gb"] = None


def temporal_offsets(num_refs: int, mode: str, step: int) -> list[int]:
    if num_refs <= 0:
        return []
    if mode == "duplicate":
        return [0] * num_refs
    if mode == "previous":
        return [-(index + 1) * step for index in range(num_refs)]
    offsets: list[int] = []
    radius = 1
    while len(offsets) < num_refs:
        offsets.extend([-radius * step, radius * step])
        radius += 1
    return offsets[:num_refs]


def load_dataset_clips(args: argparse.Namespace) -> list[tuple[list[Path], tuple[int, int]]]:
    assert args.dataset_root is not None
    annotation_path = args.dataset_root / "annotations" / f"instances_{args.split}2017.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_dir = args.dataset_root / f"{args.split}2017"
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in data.get("images", []):
        video_id = str(image.get("video_id", Path(str(image["file_name"])).stem.rsplit("_", 1)[0]))
        by_video[video_id].append(image)

    offsets = temporal_offsets(
        args.temporal_num_ref_frames,
        args.temporal_ref_frame_mode,
        args.temporal_ref_frame_step,
    )
    clips: list[tuple[list[Path], tuple[int, int]]] = []
    for video_id in sorted(by_video):
        frames = sorted(
            by_video[video_id],
            key=lambda item: (int(item.get("frame_id", 0)), int(item["id"])),
        )
        for index, image in enumerate(frames):
            paths = [image_dir / str(image["file_name"])]
            paths.extend(
                image_dir / str(frames[min(max(index + offset, 0), len(frames) - 1)]["file_name"])
                for offset in offsets
            )
            clips.append((paths, (int(image["height"]), int(image["width"]))))
            if len(clips) == args.max_images:
                break
        if len(clips) == args.max_images:
            break
    missing = [path for paths, _ in clips for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset images are missing, first path: {missing[0]}")
    if not clips:
        raise RuntimeError("No images were found in the selected dataset split")
    return clips


def kept_label_ids(classes: list[str], label_space: str) -> set[int]:
    mappings = {
        "local": {"car": 1, "bus": 2, "truck": 3},
        "coco": {"car": 3, "bus": 6, "truck": 8},
    }
    unknown = sorted(set(classes) - set(mappings[label_space]))
    if unknown:
        raise ValueError(f"Unknown classes for {label_space} label space: {unknown}")
    return {mappings[label_space][name] for name in classes}


@torch.inference_mode()
def benchmark_pure(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    tensor_dtype: torch.dtype,
) -> dict[str, Any]:
    from rfdetr.utilities.tensors import NestedTensor

    channels = 3 * (args.temporal_num_ref_frames + 1)
    inputs = torch.randn(
        args.batch_size,
        channels,
        args.resolution,
        args.resolution,
        device=device,
        dtype=tensor_dtype,
    )
    mask = torch.zeros(
        args.batch_size,
        args.resolution,
        args.resolution,
        dtype=torch.bool,
        device=device,
    )
    samples = NestedTensor(inputs, mask)
    forward = lambda: model(samples)

    reset_cuda_memory(device)
    run_warmup(forward, args.warmup, device)
    if device.type == "cuda":
        times_ms = measure_cuda_forward(forward, args.iters, device)
        timer = "torch.cuda.Event"
    else:
        times_ms = measure_wall_clock(forward, args.iters, device)
        timer = "time.perf_counter"

    result: dict[str, Any] = {
        "benchmark_type": "pure_model",
        "timer": timer,
        "input_shape": list(inputs.shape),
        **latency_metrics(times_ms, args.batch_size),
    }
    add_peak_memory(result, device)
    return result


def image_groups(args: argparse.Namespace) -> list[list[Path]]:
    assert args.input_images is not None
    frames_per_clip = args.temporal_num_ref_frames + 1
    if len(args.input_images) == frames_per_clip:
        return [list(args.input_images) for _ in range(args.batch_size)]
    return [
        list(args.input_images[offset : offset + frames_per_clip])
        for offset in range(0, len(args.input_images), frames_per_clip)
    ]


def read_and_preprocess_clip(
    paths: list[Path],
    resolution: int,
    means: list[float],
    stds: list[float],
) -> tuple[torch.Tensor, tuple[int, int]]:
    frames: list[torch.Tensor] = []
    key_size: tuple[int, int] | None = None
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if index == 0:
                key_size = (image.height, image.width)
            frame = transform.pil_to_tensor(image).float().div_(255.0)
        frame = transform.resize(frame, [resolution, resolution], antialias=True)
        frames.append(transform.normalize(frame, means, stds))
    assert key_size is not None
    return torch.cat(frames, dim=0), key_size


@torch.inference_mode()
def benchmark_e2e(
    model: torch.nn.Module,
    wrapper: Any,
    args: argparse.Namespace,
    device: torch.device,
    tensor_dtype: torch.dtype,
) -> dict[str, Any]:
    groups = image_groups(args)
    means = [float(value) for value in wrapper.means]
    stds = [float(value) for value in wrapper.stds]
    postprocess = wrapper.model.postprocess

    def pipeline() -> Any:
        clips: list[torch.Tensor] = []
        key_sizes: list[tuple[int, int]] = []
        for paths in groups:
            clip, key_size = read_and_preprocess_clip(paths, args.resolution, means, stds)
            clips.append(clip)
            key_sizes.append(key_size)
        packed = torch.stack(clips).to(device=device, dtype=tensor_dtype)
        outputs = model(packed)
        target_sizes = torch.tensor(key_sizes, dtype=torch.int64, device=device)
        return postprocess(outputs, target_sizes=target_sizes)

    reset_cuda_memory(device)
    run_warmup(pipeline, args.warmup, device)
    times_ms = measure_wall_clock(pipeline, args.iters, device)
    result: dict[str, Any] = {
        "benchmark_type": "end_to_end",
        "timer": "time.perf_counter_with_cuda_synchronize",
        "input_shape": [
            args.batch_size,
            3 * (args.temporal_num_ref_frames + 1),
            args.resolution,
            args.resolution,
        ],
        "includes": ["image_read", "resize_normalize", "clip_assembly", "forward", "postprocess"],
        **latency_metrics(times_ms, args.batch_size),
    }
    add_peak_memory(result, device)
    return result


@torch.inference_mode()
def benchmark_dataset(
    model: torch.nn.Module,
    wrapper: Any,
    args: argparse.Namespace,
    device: torch.device,
    tensor_dtype: torch.dtype,
) -> dict[str, Any]:
    clips = load_dataset_clips(args)
    means = [float(value) for value in wrapper.means]
    stds = [float(value) for value in wrapper.stds]
    postprocess = wrapper.model.postprocess
    keep_labels = kept_label_ids(args.classes, args.label_space)

    def prepare(paths: list[Path]) -> torch.Tensor:
        clip, _ = read_and_preprocess_clip(paths, args.resolution, means, stds)
        return clip.unsqueeze(0).to(device=device, dtype=tensor_dtype)

    def filter_predictions(outputs: Any, key_size: tuple[int, int]) -> int:
        target_sizes = torch.tensor([key_size], dtype=torch.int64, device=device)
        result = postprocess(outputs, target_sizes=target_sizes)[0]
        labels = result["labels"]
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for label_id in keep_labels:
            keep |= labels == label_id
        return int(keep.sum().item())

    warmup_input = prepare(clips[0][0])
    for _ in range(args.warmup):
        model(warmup_input)
    synchronize(device)
    reset_cuda_memory(device)

    times_ms: list[float] = []
    kept_predictions = 0
    for index, (paths, key_size) in enumerate(clips):
        packed = prepare(paths)
        synchronize(device)
        started = time.perf_counter()
        outputs = model(packed)
        synchronize(device)
        times_ms.append((time.perf_counter() - started) * 1000.0)
        kept_predictions += filter_predictions(outputs, key_size)
        if (index + 1) % 25 == 0 or index + 1 == len(clips):
            print(f"RF-DETR benchmark: {index + 1}/{len(clips)} key frames", flush=True)

    result: dict[str, Any] = {
        "benchmark_type": "dataset_model_forward",
        "timer": "time.perf_counter_with_device_synchronize",
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "measured_key_images": len(clips),
        "input_frames_per_key_image": args.temporal_num_ref_frames + 1,
        "classes": list(args.classes),
        "label_space": args.label_space,
        "kept_predictions": kept_predictions,
        "includes": ["forward"],
        "excludes": [
            "image_read",
            "resize_normalize",
            "host_to_device",
            "bbox_postprocess",
            "class_filter",
            "drawing",
        ],
        "total_timed_seconds": sum(times_ms) / 1000.0,
        **latency_metrics(times_ms, 1),
    }
    add_peak_memory(result, device)
    return result


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch build")

    wrapper = load_wrapper(args)
    model = get_core_model(wrapper).to(device).eval()
    tensor_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model.to(dtype=tensor_dtype)

    common: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "variant": args.variant,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "backbone_lora": args.backbone_lora,
        "temporal_num_ref_frames": args.temporal_num_ref_frames,
        "temporal_key_frame_position": args.temporal_key_frame_position,
        "temporal_tdte_layers": args.temporal_tdte_layers,
        "temporal_tdtd_layers": args.temporal_tdtd_layers,
        "evaluated_classes": list(args.classes),
        "predictions_per_clip": 1,
        "fps_unit": "key_frames_per_second",
        "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        common["gpu_name"] = torch.cuda.get_device_name(device)

    if args.benchmark_mode == "pure":
        return {**common, **benchmark_pure(model, args, device, tensor_dtype)}
    if args.benchmark_mode == "e2e":
        return {**common, **benchmark_e2e(model, wrapper, args, device, tensor_dtype)}
    if args.benchmark_mode == "dataset":
        return {**common, **benchmark_dataset(model, wrapper, args, device, tensor_dtype)}
    return {
        **common,
        "benchmark_type": "both",
        "pure_model": benchmark_pure(model, args, device, tensor_dtype),
        "end_to_end": benchmark_e2e(model, wrapper, args, device, tensor_dtype),
    }


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
