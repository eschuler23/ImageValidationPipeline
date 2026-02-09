#!/usr/bin/env python3
"""
Content-validation pipeline entry point.

Design goals
- Keep this file extremely small and readable.
- Do *only* the initial loading of images + ground truth here.
- Call well-named functions from other modules for every other step.

If you want to understand the pipeline, start here and then open the
module for each step that is called below.
"""
from __future__ import annotations

from pathlib import Path

import train_binary_classifier as tbc
from pipeline_steps.loading import load_image_root
from pipeline_steps.preprocessing import (
    build_augmentation_settings,
    build_label_texts,
    build_variants,
    build_row_lookup,
    build_split_label_report,
    build_stratify_keys,
    load_ground_truth_and_match,
    resolve_report_columns,
    resolve_stratify_columns,
    split_samples,
)
from pipeline_steps.reporting import (
    init_run_root,
    write_data_stats,
    write_dataset_manifest,
    write_preprocessing,
    write_split_report,
    write_summary,
)
from pipeline_steps.training import train_models


def _validate_image_dir(image_dir: Path) -> Path:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image dir is not a directory: {image_dir}")
    return image_dir


def main() -> None:
    args = tbc.build_arg_parser().parse_args()
    args = tbc.prepare_args(args)
    if len(args.models) != 1:
        raise ValueError("main.py runs exactly one model per run. Pass a single --models entry.")

    # 1) Initial loading of images.
    if args.image_root is not None:
        load_image_root(args.image_root)
    else:
        _validate_image_dir(args.image_dir)

    # 2) Load ground truth + match CSV rows to on-disk images.
    ground_truth_rows, ground_truth_stats, samples, match_stats = load_ground_truth_and_match(
        args
    )

    by_project_filename, by_filename = build_row_lookup(ground_truth_rows)
    stratify_columns = resolve_stratify_columns(args, ground_truth_rows)
    stratify_keys = (
        build_stratify_keys(
            samples,
            by_project_filename=by_project_filename,
            by_filename=by_filename,
            columns=stratify_columns,
            label_column=args.label_column,
        )
        if args.split_strategy == "stratified"
        else None
    )

    tbc.set_seed(args.seed)

    train_samples, val_samples, test_samples = split_samples(
        samples,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        split_strategy=args.split_strategy,
        stratify_keys=stratify_keys,
    )
    augment_settings = build_augmentation_settings(args)
    train_variants, val_variants, test_variants = build_variants(
        train_samples,
        val_samples,
        test_samples,
        augment_settings,
    )
    label_texts = build_label_texts(args.positive_labels, args.negative_labels)

    # 4) Initialize run output and write preprocessing + dataset stats.
    run_root = init_run_root(args.run_root, args.run_name)
    report_columns = resolve_report_columns(args, ground_truth_rows)
    split_report = build_split_label_report(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        by_project_filename=by_project_filename,
        by_filename=by_filename,
        columns=report_columns,
        label_column=args.label_column,
    )
    match_stats.update(
        {
            "ground_truth_rows": len(ground_truth_rows),
            "ground_truth_missing_project": ground_truth_stats.get("missing_project", 0),
            "ground_truth_missing_filename": ground_truth_stats.get("missing_filename", 0),
            "ground_truth_missing_label": ground_truth_stats.get("missing_label", 0),
            "ground_truth_decoded_percent_newlines": ground_truth_stats.get(
                "decoded_percent_newlines", 0
            ),
        }
    )
    write_data_stats(
        run_root=run_root,
        stats=match_stats,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_variants=train_variants,
        val_variants=val_variants,
        test_variants=test_variants,
    )
    write_preprocessing(
        run_root=run_root,
        args=args,
        train_variants=train_variants,
        val_variants=val_variants,
        test_variants=test_variants,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )
    write_dataset_manifest(
        run_root=run_root,
        train_variants=train_variants,
        val_variants=val_variants,
        test_variants=test_variants,
        label_texts=label_texts,
    )
    write_split_report(run_root, split_report)

    # 5) Train and evaluate models.
    summary = train_models(
        args=args,
        train_variants=train_variants,
        val_variants=val_variants,
        test_variants=test_variants,
        label_texts=label_texts,
        run_root=run_root,
    )
    write_summary(run_root, summary)

    print("\nSummary:")
    for row in summary:
        print(
            f"- {row['model']}: val_f1={row['val_f1']:.4f} "
            f"test_f1={row['test_f1']:.4f} "
            f"test_acc={row['test_accuracy']:.4f}"
        )
    print(f"Outputs saved to: {run_root}")


if __name__ == "__main__":
    main()
