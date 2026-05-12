"""
Reporting utilities for the content-validation pipeline.

These helpers write the run artifacts in a consistent, human-readable format.
They are intentionally simple so the main pipeline stays readable and the
reporting format remains stable across runs.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import train_binary_classifier as tbc


def init_run_root(run_root: Path, run_name: str | None) -> Path:
    """Create and return the run output directory."""

    resolved_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    resolved_root = run_root / resolved_name
    resolved_root.mkdir(parents=True, exist_ok=True)
    return resolved_root


def write_data_stats(
    *,
    run_root: Path,
    stats: Mapping[str, int],
    train_samples: Sequence[tbc.Sample],
    val_samples: Sequence[tbc.Sample],
    test_samples: Sequence[tbc.Sample],
    train_variants: Sequence[tbc.SampleVariant],
    val_variants: Sequence[tbc.SampleVariant],
    test_variants: Sequence[tbc.SampleVariant],
) -> None:
    """Persist dataset counts and augmentation breakdown."""

    payload: Dict[str, object] = {
        **stats,
        "samples": len(train_samples) + len(val_samples) + len(test_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_variants": len(train_variants),
        "val_variants": len(val_variants),
        "test_variants": len(test_variants),
        "train_augmentation_counts": dict(
            Counter(v.augmentation for v in train_variants)
        ),
    }

    with (run_root / "data_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_preprocessing(
    *,
    run_root: Path,
    args,
    train_variants: Sequence[tbc.SampleVariant],
    val_variants: Sequence[tbc.SampleVariant],
    test_variants: Sequence[tbc.SampleVariant],
    train_samples: Sequence[tbc.Sample],
    val_samples: Sequence[tbc.Sample],
    test_samples: Sequence[tbc.Sample],
) -> None:
    """Capture preprocessing and augmentation settings for reproducibility."""

    preprocessing_payload = {
        "csv_path": str(args.csv_path),
        "image_dir": str(args.image_dir) if args.image_dir else None,
        "image_root": str(args.image_root) if args.image_root else None,
        "project_column": args.project_column if args.image_root else None,
        "filename_column": args.filename_column,
        "label_column": args.label_column,
        "positive_labels": args.positive_labels,
        "negative_labels": args.negative_labels,
        "project_map": str(args.project_map) if args.project_map else None,
        "decode_percent_newlines": args.decode_percent_newlines,
        "dedupe_filenames": args.dedupe_filenames,
        "random_augmentation": args.augment,
        "augmentation": {
            "random_hflip": args.augment_random_hflip,
            "random_vflip": args.augment_random_vflip,
            "random_rotations": list(args.augment_random_rotations),
            "random_rotation_prob": args.augment_random_rotation_prob,
            "random_blur": args.augment_random_blur,
            "random_blur_prob": args.augment_random_blur_prob,
            "random_blur_candidates": list(
                getattr(args, "augment_random_blur_candidates", ())
            ),
            "random_blur_range": (
                list(args.augment_random_blur_range)
                if args.augment_random_blur_range is not None
                else None
            ),
            "random_blur_reject_radius": args.augment_random_blur_reject_radius,
            "blur_keep": list(args.augment_blur_keep),
            "blur_flip": list(args.augment_blur_flip),
            "blur_only_positive": args.augment_blur_only_positive,
            "blur_size_aware": args.augment_blur_size_aware,
            "blur_size_small_max_mp": args.blur_size_small_max_mp,
            "blur_size_medium_max_mp": args.blur_size_medium_max_mp,
            "blur_switch_small": args.blur_switch_small,
            "blur_switch_medium": args.blur_switch_medium,
            "blur_switch_large": args.blur_switch_large,
            "balance_train_target": args.balance_train_target,
            "balance_train_seed": args.balance_train_seed,
            "crop_jitter": args.augment_crop_jitter,
            "jpeg_quality": args.augment_jpeg_quality,
            "jpeg_prob": args.augment_jpeg_prob,
            "noise_std": args.augment_noise_std,
            "noise_prob": args.augment_noise_prob,
            "color_jitter": args.augment_color_jitter,
            "color_jitter_hue": args.augment_color_jitter_hue,
            "color_jitter_saturation": args.augment_color_jitter_saturation,
            "color_jitter_contrast": args.augment_color_jitter_contrast,
            "brightness_jitter": args.augment_brightness_jitter,
            "brightness_jitter_delta": args.augment_brightness_jitter_delta,
        },
        "training": {
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "unfreeze_lr": args.unfreeze_lr,
            "weight_decay": args.weight_decay,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "unfreeze_last_block": args.unfreeze_last_block,
            "freeze_batchnorm": args.freeze_batchnorm,
            "amp": args.amp,
        },
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "splits": {
            "train_base": len(train_samples),
            "val_base": len(val_samples),
            "test_base": len(test_samples),
        },
        "variants": {
            "train": len(train_variants),
            "val": len(val_variants),
            "test": len(test_variants),
        },
    }

    with (run_root / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessing_payload, handle, indent=2)


def write_dataset_manifest(
    *,
    run_root: Path,
    train_variants: Sequence[tbc.SampleVariant],
    val_variants: Sequence[tbc.SampleVariant],
    test_variants: Sequence[tbc.SampleVariant],
    label_texts: Dict[int, str],
) -> None:
    """Save a per-sample manifest of splits and augmentations."""

    tbc.write_dataset_manifest(
        run_root / "dataset_manifest.csv",
        splits={
            "train": train_variants,
            "val": val_variants,
            "test": test_variants,
        },
        label_texts=label_texts,
    )


def write_summary(run_root: Path, summary: Sequence[tbc.SummaryRow]) -> None:
    """Persist the summary metrics for all models."""

    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def write_split_report(run_root: Path, report: Dict[str, object]) -> None:
    """Persist split balance summaries for label columns."""

    with (run_root / "split_label_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
