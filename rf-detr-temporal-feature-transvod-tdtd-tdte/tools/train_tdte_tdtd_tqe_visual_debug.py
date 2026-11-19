from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = REPO_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from rfdetr._namespace import _namespace_from_configs  # noqa: E402
from rfdetr.config import RFDETRSmallConfig, TrainConfig  # noqa: E402
from rfdetr.datasets import build_dataset  # noqa: E402
from rfdetr.models import build_criterion_from_config, build_model_from_config  # noqa: E402
from rfdetr.utilities import box_ops  # noqa: E402
from rfdetr.utilities.tensors import make_collate_fn  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TDTE+TDTD+TQE training for a short time and save clip/key-frame visualizations."
    )
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "car_dataset_temporal"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "result" / "tdte_tdtd_tqe_visual_debug"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-max-norm", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--max-preds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_batch_to_device(batch: tuple[Any, Any], device: torch.device) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    samples, targets = batch
    non_blocking = device.type == "cuda"
    samples = samples.to(device, non_blocking=non_blocking)
    targets = [{key: value.to(device, non_blocking=non_blocking) for key, value in target.items()} for target in targets]
    return samples, targets


def weighted_loss(loss_dict: dict[str, torch.Tensor], weight_dict: dict[str, float]) -> torch.Tensor:
    return sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    tensor = (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def target_boxes_xyxy(target: dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = target["boxes"].detach().cpu()
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    size = target["size"].detach().cpu()
    height, width = float(size[0]), float(size[1])
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes * scale


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: list[str],
    *,
    color: str,
    title: str,
    note: str | None = None,
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width - 1, 15), fill=(0, 0, 0))
    draw.text((4, 3), title, fill="white", font=font)
    if note:
        draw.text((4, image.height - 13), note, fill="white", font=font, stroke_width=2, stroke_fill="black")
    for box, label in zip(boxes.tolist(), labels):
        x1, y1, x2, y2 = box
        x1 = max(0, min(image.width - 1, x1))
        y1 = max(0, min(image.height - 1, y1))
        x2 = max(0, min(image.width - 1, x2))
        y2 = max(0, min(image.height - 1, y2))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 2, max(16, y1 + 2)), label, fill=color, font=font, stroke_width=2, stroke_fill="black")
    return image


def save_side_by_side(left: Image.Image, right: Image.Image, path: Path) -> None:
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(path)


def build_training_objects(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, Any, DataLoader, Any]:
    model_config = RFDETRSmallConfig(
        pretrain_weights=None,
        num_classes=args.num_classes,
        resolution=args.resolution,
        amp=False,
        compile=False,
        temporal_num_ref_frames=3,
        temporal_fusion_layers=1,
        temporal_tdte_layers=1,
        temporal_tdtd_layers=1,
        temporal_key_frame_position="center",
        device=str(device),
    )
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        epochs=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_max_norm=args.clip_max_norm,
        num_workers=args.num_workers,
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=False,
        augmentation_backend="cpu",
        use_ema=False,
        tensorboard=False,
        temporal_ref_frame_mode="surrounding",
        temporal_key_frame_position="center",
        seed=args.seed,
    )
    namespace = _namespace_from_configs(model_config, train_config)
    dataset = build_dataset("train", namespace, model_config.resolution)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate_fn(block_size=model_config.patch_size * model_config.num_windows),
    )
    model = build_model_from_config(model_config, train_config).to(device)
    criterion, postprocess = build_criterion_from_config(model_config, train_config)
    criterion = criterion.to(device)
    return model_config, train_config, model, loader, (criterion, postprocess)


def save_visuals(
    *,
    samples: Any,
    targets: list[dict[str, torch.Tensor]],
    outputs: dict[str, torch.Tensor],
    postprocess: Any,
    output_dir: Path,
    clip_index: int,
    score_threshold: float,
    max_preds: int,
) -> None:
    clips_dir = output_dir / "clips"
    comparison_dir = output_dir / "comparison"
    clips_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    tensors = samples.tensors.detach().cpu()
    batch_size, channels, _, _ = tensors.shape
    num_frames = channels // 3
    key_frame_index = num_frames // 2
    target_sizes = torch.stack([target["size"] for target in targets]).to(outputs["pred_logits"].device)
    predictions = postprocess(outputs, target_sizes)

    for batch_index in range(batch_size):
        clip_name = f"clip{clip_index:03d}_b{batch_index + 1}"
        frame_tensors = tensors[batch_index].view(num_frames, 3, tensors.shape[-2], tensors.shape[-1])
        gt_boxes = target_boxes_xyxy(targets[batch_index])
        gt_labels = [f"gt:{int(label)}" for label in targets[batch_index]["labels"].detach().cpu().tolist()]

        key_before: Image.Image | None = None
        key_after: Image.Image | None = None
        ref_counter = 1
        for frame_index, frame_tensor in enumerate(frame_tensors):
            role = "key" if frame_index == key_frame_index else f"ref{ref_counter}"
            if frame_index != key_frame_index:
                ref_counter += 1
            image = tensor_to_image(frame_tensor)
            if frame_index == key_frame_index:
                image_with_boxes = draw_boxes(
                    image,
                    gt_boxes,
                    gt_labels,
                    color="lime",
                    title=f"{clip_name}_{role}: key GT",
                )
                key_before = image_with_boxes
            else:
                image_with_boxes = draw_boxes(
                    image,
                    gt_boxes,
                    gt_labels,
                    color="orange",
                    title=f"{clip_name}_{role}",
                    note="reference frame: key-frame GT overlay; ref GT is unavailable",
                )
            image_with_boxes.save(clips_dir / f"{clip_name}_{role}.png")

        prediction = predictions[batch_index]
        keep = prediction["scores"].detach().cpu() >= score_threshold
        pred_boxes = prediction["boxes"].detach().cpu()[keep][:max_preds]
        pred_scores = prediction["scores"].detach().cpu()[keep][:max_preds]
        pred_labels = prediction["labels"].detach().cpu()[keep][:max_preds]
        pred_text = [f"pred:{int(label)} {score:.2f}" for label, score in zip(pred_labels.tolist(), pred_scores.tolist())]
        if key_before is not None:
            key_raw = tensor_to_image(frame_tensors[key_frame_index])
            key_after = draw_boxes(
                key_raw,
                pred_boxes,
                pred_text,
                color="red",
                title=f"{clip_name}_key: after TDTE+TDTD+TQE",
                note=f"score >= {score_threshold:.2f}; {len(pred_boxes)} boxes",
            )
            key_before.save(comparison_dir / f"{clip_name}_key_before_gt.png")
            key_after.save(comparison_dir / f"{clip_name}_key_after_pred.png")
            save_side_by_side(key_before, key_after, comparison_dir / f"{clip_name}_key_side_by_side.png")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    model_config, train_config, model, loader, criterion_and_postprocess = build_training_objects(args, device)
    criterion, postprocess = criterion_and_postprocess
    model.train()
    criterion.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        fused=False,
    )

    log_path = output_dir / "train_steps.jsonl"
    started = time.perf_counter()
    data_iter = iter(loader)
    step = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.perf_counter() - started
            if step >= args.min_steps and elapsed >= args.seconds:
                break

            data_started = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            data_time_ms = (time.perf_counter() - data_started) * 1000.0

            samples, targets = move_batch_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            step_started = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(samples, targets)
            save_outputs = {key: value.detach() if torch.is_tensor(value) else value for key, value in outputs.items()}
            loss_dict = criterion(outputs, targets)
            loss = weighted_loss(loss_dict, criterion.weight_dict)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_max_norm)
            optimizer.step()

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize(device)
                compute_gpu_ms = float(start_event.elapsed_time(end_event))
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            else:
                compute_gpu_ms = (time.perf_counter() - step_started) * 1000.0
                peak_vram_gb = 0.0
            step_wall_ms = (time.perf_counter() - step_started) * 1000.0

            save_visuals(
                samples=samples,
                targets=targets,
                outputs=save_outputs,
                postprocess=postprocess,
                output_dir=output_dir,
                clip_index=step + 1,
                score_threshold=args.score_threshold,
                max_preds=args.max_preds,
            )

            record = {
                "step": step,
                "elapsed_s": time.perf_counter() - started,
                "input_shape": list(samples.tensors.shape),
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "data_time_ms": data_time_ms,
                "compute_gpu_ms": compute_gpu_ms,
                "step_wall_ms": step_wall_ms,
                "key_fps": args.batch_size * 1000.0 / compute_gpu_ms,
                "raw_frame_fps": args.batch_size * 4 * 1000.0 / compute_gpu_ms,
                "peak_vram_gb": peak_vram_gb,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                f"step={step + 1} elapsed={record['elapsed_s']:.1f}s loss={record['loss']:.4f} "
                f"compute={compute_gpu_ms:.1f}ms key_fps={record['key_fps']:.2f} peak_vram={peak_vram_gb:.2f}GB",
                flush=True,
            )
            step += 1

    summary = {
        "variant": "small",
        "mode": "tdte_tdtd_tqe",
        "steps": step,
        "elapsed_s": time.perf_counter() - started,
        "input_shape": [args.batch_size, 12, args.resolution, args.resolution],
        "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "output_dir": str(output_dir),
        "clip_dir": str(output_dir / "clips"),
        "comparison_dir": str(output_dir / "comparison"),
        "log_path": str(log_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote visual debug output to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
