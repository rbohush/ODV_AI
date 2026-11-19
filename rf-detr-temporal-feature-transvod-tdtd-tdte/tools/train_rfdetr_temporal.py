#!/usr/bin/env python3
"""Train temporal RF-DETR on COCO frame sequences.

The temporal model reads the key frame plus reference frames selected from the
same ``video_id`` sequence.  The default mode is ``previous`` so evaluation stays
causal for driving/autopilot use: detections for frame T never use future frames.
"""

from __future__ import annotations

'''
этот скрипт запускает rf-detr с temporal режимом.
он передает модели текущий кадр и выбранные соседние кадры, чтобы temporal
fusion мог сравнить их query и улучшить устойчивость детекции на видео.
'''

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal RF-DETR on COCO frame sequences.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "data" / "dandelion_dataset2_temporal"),
        help="Dataset root with train2017/val2017/annotations containing video_id/frame_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "exps" / "rfdetr_temporal_prev3"),
        help="Training output directory.",
    )
    parser.add_argument(
        "--variant",
        choices=["nano", "small", "medium", "large"],
        default="small",
        help="RF-DETR model size.",
    )
    parser.add_argument("--num-classes", type=int, default=1, help="Number of foreground classes.")
    parser.add_argument("--resolution", type=int, default=640, help="Training/inference resize resolution.")
    parser.add_argument(
        "--backbone-lora",
        action="store_true",
        help="Enable RF-DETR LoRA/DoRA adapters in the DINOv2 backbone.",
    )
    parser.add_argument(
        "--freeze-rfdetr-base",
        action="store_true",
        help="Freeze base RF-DETR weights; train only temporal modules and enabled LoRA/DoRA adapters.",
    )
    parser.add_argument(
        "--temporal-num-ref-frames",
        type=int,
        default=3,
        help="Number of reference frames packed with every key frame.",
    )
    parser.add_argument(
        "--temporal-fusion-layers",
        type=int,
        default=1,
        help="Number of query-fusion layers applied after the RF-DETR decoder.",
    )
    parser.add_argument(
        "--temporal-tdtd-layers",
        type=int,
        default=0,
        help="Number of TransVOD-style TDTD layers after temporal query fusion. 0 disables TDTD.",
    )
    parser.add_argument(
        "--temporal-tdte-layers",
        type=int,
        default=0,
        help="Number of TransVOD-style TDTE layers before TDTD. 0 disables TDTE.",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["previous", "surrounding", "duplicate"],
        default="previous",
        help="'previous' is causal and recommended for autopilot testing.",
    )
    parser.add_argument(
        "--temporal-ref-frame-step",
        type=int,
        default=1,
        help="Step between selected reference frames.",
    )
    parser.add_argument(
        "--temporal-key-frame-position",
        choices=["first", "center"],
        default="first",
        help="'center' builds chronological TRF-DETR clips with the annotated key frame in the middle.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-drop", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None, help="Optional RF-DETR checkpoint to resume from.")
    parser.add_argument(
        "--pretrain-weights",
        default=None,
        help=(
            "Optional pretrain weights passed to RF-DETR model construction. "
            "Use 'none' to disable model pretrain weights."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--run-test", action="store_true", help="Evaluate test split after training if available.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA to reduce memory use.")
    parser.add_argument(
        "--performance-log",
        action="store_true",
        help="Write training FPS/VRAM/step-time logs and plots.",
    )
    parser.add_argument(
        "--performance-log-dir",
        default=None,
        help="Optional directory for performance.csv, performance_summary.json, and plots.",
    )
    parser.add_argument(
        "--performance-log-every-n-steps",
        type=int,
        default=1,
        help="Log one performance record every N training batches.",
    )
    parser.add_argument(
        "--realtime-fps-target",
        type=float,
        default=30.0,
        help="Target FPS used to compute realtime_ratio metrics.",
    )
    parser.add_argument(
        "--progress-bar",
        choices=["tqdm", "rich", "none"],
        default="tqdm",
        help="Training progress bar style.",
    )
    return parser.parse_args()


def get_model_class(variant: str):
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    return {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }[variant]


def normalized_pretrain_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "false", "0"}:
        return None
    return value


def main() -> None:
    args = parse_args()
    model_cls = get_model_class(args.variant)

    model_kwargs = {
        "num_classes": args.num_classes,
        "resolution": args.resolution,
        "temporal_num_ref_frames": args.temporal_num_ref_frames,
        "temporal_fusion_layers": args.temporal_fusion_layers,
        "temporal_key_frame_position": args.temporal_key_frame_position,
        "temporal_tdte_layers": args.temporal_tdte_layers,
        "temporal_tdtd_layers": args.temporal_tdtd_layers,
        "temporal_dropout": args.temporal_dropout,
        "backbone_lora": args.backbone_lora,
    }
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = normalized_pretrain_weights(args.pretrain_weights)

    model = model_cls(**model_kwargs)
    model.train(
        dataset_file="coco",
        dataset_dir=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_drop=args.lr_drop,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        use_ema=not args.no_ema,
        run_test=args.run_test,
        performance_log=args.performance_log,
        performance_log_dir=args.performance_log_dir,
        performance_log_every_n_steps=args.performance_log_every_n_steps,
        realtime_fps_target=args.realtime_fps_target,
        freeze_rfdetr_base=args.freeze_rfdetr_base,
        progress_bar=None if args.progress_bar == "none" else args.progress_bar,
        temporal_ref_frame_mode=args.temporal_ref_frame_mode,
        temporal_ref_frame_step=args.temporal_ref_frame_step,
        temporal_key_frame_position=args.temporal_key_frame_position,
        notes={
            "run_type": "rfdetr_temporal",
            "dataset_root": args.dataset_root,
            "temporal": True,
            "temporal_num_ref_frames": args.temporal_num_ref_frames,
            "temporal_fusion_layers": args.temporal_fusion_layers,
            "temporal_key_frame_position": args.temporal_key_frame_position,
            "temporal_tdte_layers": args.temporal_tdte_layers,
            "temporal_tdtd_layers": args.temporal_tdtd_layers,
            "temporal_ref_frame_mode": args.temporal_ref_frame_mode,
            "temporal_ref_frame_step": args.temporal_ref_frame_step,
            "backbone_lora": args.backbone_lora,
            "freeze_rfdetr_base": args.freeze_rfdetr_base,
        },
    )


if __name__ == "__main__":
    main()
