"""
Preprocessing step for the content-validation pipeline.

This module adapts the existing training utilities into small, readable
functions that the main pipeline can call in sequence. The goal is clarity:
- load & match samples
- split into train/val/test
 - expand deterministic blur variants
- build label text helpers
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import train_binary_classifier as tbc
from pipeline_steps.loading import GroundTruthRow, load_ground_truth_rows


def prepare_samples(args) -> Tuple[List[tbc.Sample], Dict[str, int]]:
    """
    Load CSV rows, match them to on-disk images, and return valid samples.

    This delegates to the existing, battle-tested loader in train_binary_classifier,
    so we keep matching behavior consistent with training.
    """

    return tbc.load_samples(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        image_root=args.image_root,
        label_column=args.label_column,
        positive_labels=args.positive_labels,
        negative_labels=args.negative_labels,
        filename_column=args.filename_column,
        project_column=args.project_column,
        project_map=tbc.load_project_map(args.project_map),
        decode_percent_newlines=args.decode_percent_newlines,
        dedupe_filenames=args.dedupe_filenames,
        max_samples=args.max_samples,
    )


def build_row_lookup(
    rows: Sequence[GroundTruthRow],
) -> Tuple[Dict[Tuple[str, str], GroundTruthRow], Dict[str, GroundTruthRow]]:
    """
    Build lookup tables for ground-truth rows.

    We key by (project, filename) first. If a filename is unique across projects,
    we also store a filename-only lookup as a fallback.
    """

    by_project_filename: Dict[Tuple[str, str], GroundTruthRow] = {}
    filename_counts: Counter[str] = Counter()
    for row in rows:
        key = (row.project, row.filename)
        by_project_filename[key] = row
        if row.filename:
            filename_counts[row.filename] += 1

    by_filename: Dict[str, GroundTruthRow] = {}
    for row in rows:
        if row.filename and filename_counts[row.filename] == 1:
            by_filename[row.filename] = row

    return by_project_filename, by_filename


def _normalize_value(value: str) -> str:
    cleaned = value.strip().lower().replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or "<missing>"


def _row_for_sample(
    sample: tbc.Sample,
    by_project_filename: Mapping[Tuple[str, str], GroundTruthRow],
    by_filename: Mapping[str, GroundTruthRow],
) -> GroundTruthRow | None:
    if sample.project:
        key = (sample.project, sample.filename)
        if key in by_project_filename:
            return by_project_filename[key]
    return by_filename.get(sample.filename)


def resolve_stratify_columns(
    args,
    rows: Sequence[GroundTruthRow],
) -> List[str]:
    if args.stratify_columns:
        return list(args.stratify_columns)

    columns: set[str] = set()
    for row in rows:
        columns.update(row.raw_row.keys())

    stratify = [args.label_column]
    if "Blurr" in columns and "Blurr" not in stratify:
        stratify.append("Blurr")
    return stratify


def resolve_report_columns(
    args,
    rows: Sequence[GroundTruthRow],
) -> List[str]:
    if args.stratify_report_columns:
        return list(args.stratify_report_columns)

    columns: set[str] = set()
    for row in rows:
        columns.update(row.raw_row.keys())
    columns.discard("filename")
    columns.discard("project")
    if args.label_column in columns:
        columns.remove(args.label_column)
        return [args.label_column] + sorted(columns)
    return sorted(columns)


def build_stratify_keys(
    samples: Sequence[tbc.Sample],
    *,
    by_project_filename: Mapping[Tuple[str, str], GroundTruthRow],
    by_filename: Mapping[str, GroundTruthRow],
    columns: Sequence[str],
    label_column: str,
) -> List[Tuple[str, ...]]:
    keys: List[Tuple[str, ...]] = []
    for sample in samples:
        row = _row_for_sample(sample, by_project_filename, by_filename)
        values: List[str] = []
        for column in columns:
            if column == label_column:
                raw_value = sample.label_raw
            else:
                raw_value = row.raw_row.get(column, "") if row else ""
            values.append(_normalize_value(raw_value))
        keys.append(tuple(values))
    return keys


def stratified_split_samples(
    samples: Sequence[tbc.Sample],
    *,
    stratify_keys: Sequence[Tuple[str, ...]],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[tbc.Sample], List[tbc.Sample], List[tbc.Sample]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, key in enumerate(stratify_keys):
        groups[key].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for key in sorted(groups.keys()):
        indices = groups[key]
        rng.shuffle(indices)
        total = len(indices)

        test_count = int(round(total * test_split))
        val_count = int(round(total * val_split))

        if total >= 3:
            if test_count == 0:
                test_count = 1
            if val_count == 0:
                val_count = 1
        elif total == 2 and val_count == 0 and test_count == 0:
            val_count = 1

        while test_count + val_count >= total and (test_count > 0 or val_count > 0):
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1

        test_idx.extend(indices[:test_count])
        val_idx.extend(indices[test_count : test_count + val_count])
        train_idx.extend(indices[test_count + val_count :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples


def _summarize_split(
    samples: Sequence[tbc.Sample],
    *,
    by_project_filename: Mapping[Tuple[str, str], GroundTruthRow],
    by_filename: Mapping[str, GroundTruthRow],
    columns: Sequence[str],
    label_column: str,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {column: {} for column in columns}
    for sample in samples:
        row = _row_for_sample(sample, by_project_filename, by_filename)
        for column in columns:
            if column == label_column:
                raw_value = sample.label_raw
            else:
                raw_value = row.raw_row.get(column, "") if row else ""
            value = _normalize_value(raw_value)
            summary[column][value] = summary[column].get(value, 0) + 1
    return summary


def build_split_label_report(
    *,
    train_samples: Sequence[tbc.Sample],
    val_samples: Sequence[tbc.Sample],
    test_samples: Sequence[tbc.Sample],
    by_project_filename: Mapping[Tuple[str, str], GroundTruthRow],
    by_filename: Mapping[str, GroundTruthRow],
    columns: Sequence[str],
    label_column: str,
) -> Dict[str, object]:
    overall_samples = list(train_samples) + list(val_samples) + list(test_samples)
    overall = _summarize_split(
        overall_samples,
        by_project_filename=by_project_filename,
        by_filename=by_filename,
        columns=columns,
        label_column=label_column,
    )
    train = _summarize_split(
        train_samples,
        by_project_filename=by_project_filename,
        by_filename=by_filename,
        columns=columns,
        label_column=label_column,
    )
    val = _summarize_split(
        val_samples,
        by_project_filename=by_project_filename,
        by_filename=by_filename,
        columns=columns,
        label_column=label_column,
    )
    test = _summarize_split(
        test_samples,
        by_project_filename=by_project_filename,
        by_filename=by_filename,
        columns=columns,
        label_column=label_column,
    )

    missing: Dict[str, Dict[str, List[str]]] = {}
    for column in columns:
        overall_labels = set(overall.get(column, {}).keys())
        missing[column] = {
            "val": sorted(overall_labels - set(val.get(column, {}).keys())),
            "test": sorted(overall_labels - set(test.get(column, {}).keys())),
        }

    return {
        "columns": list(columns),
        "overall": overall,
        "train": train,
        "val": val,
        "test": test,
        "missing_labels": missing,
    }


def load_ground_truth_and_match(
    args,
) -> Tuple[List[GroundTruthRow], Dict[str, int], List[tbc.Sample], Dict[str, int]]:
    """
    Load ground-truth rows and match labels to on-disk images.

    This keeps label-matching logic out of the main entry point so the pipeline
    remains a high-level orchestration script.
    """

    ground_truth_rows, ground_truth_stats = load_ground_truth_rows(
        args.csv_path,
        project_column=args.project_column,
        filename_column=args.filename_column,
        label_column=args.label_column,
        decode_percent_newlines=args.decode_percent_newlines,
    )
    samples, match_stats = prepare_samples(args)
    if not samples:
        raise ValueError("No samples matched the provided label mapping.")
    return ground_truth_rows, ground_truth_stats, samples, match_stats


def split_samples(
    samples: Sequence[tbc.Sample],
    *,
    val_split: float,
    test_split: float,
    seed: int,
    split_strategy: str = "stratified",
    stratify_keys: Sequence[Tuple[str, ...]] | None = None,
) -> Tuple[List[tbc.Sample], List[tbc.Sample], List[tbc.Sample]]:
    """
    Split the dataset into train/val/test.

    When stratified, the split preserves the distribution of the provided
    stratify keys (usually the label column).
    """

    if split_strategy == "stratified":
        if stratify_keys is None or len(stratify_keys) != len(samples):
            raise ValueError("Stratified split requires stratify keys for all samples.")
        return stratified_split_samples(
            samples,
            stratify_keys=stratify_keys,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )

    train_idx, val_idx, test_idx = tbc.split_indices(len(samples), val_split, test_split, seed)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples


def build_augmentation_settings(args) -> tbc.AugmentationSettings:
    """Create the augmentation settings object used by the training pipeline."""

    return tbc.AugmentationSettings(
        blur_keep=args.augment_blur_keep,
        blur_flip=args.augment_blur_flip,
        blur_only_positive=args.augment_blur_only_positive,
        blur_size_aware=args.augment_blur_size_aware,
        blur_size_small_max_mp=args.blur_size_small_max_mp,
        blur_size_medium_max_mp=args.blur_size_medium_max_mp,
        blur_switch_small=args.blur_switch_small,
        blur_switch_medium=args.blur_switch_medium,
        blur_switch_large=args.blur_switch_large,
        balance_target=args.balance_train_target,
        balance_seed=args.balance_train_seed,
    )


def build_variants(
    train_samples: Sequence[tbc.Sample],
    val_samples: Sequence[tbc.Sample],
    test_samples: Sequence[tbc.Sample],
    augment_settings: tbc.AugmentationSettings,
) -> Tuple[List[tbc.SampleVariant], List[tbc.SampleVariant], List[tbc.SampleVariant]]:
    """Expand deterministic blur variants for the training split only."""

    train_variants = tbc.build_train_variants(train_samples, augment_settings)
    val_variants = tbc.base_variants(val_samples)
    test_variants = tbc.base_variants(test_samples)
    return train_variants, val_variants, test_variants


def build_label_texts(positive_labels: Sequence[str], negative_labels: Sequence[str]) -> Dict[int, str]:
    """Provide consistent label text for reports and manifests."""

    return tbc.build_label_texts(positive_labels, negative_labels)
