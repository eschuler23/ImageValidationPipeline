"""Compute Laplacian-variance confusion matrices for blur detection.

How to run (from repo root):
  uv run python Image_Processing/Utils/laplacian_confusion_matrices.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --label-column "usability considering blur" \
    --project-column project \
    --thresholds 100 180 \
    --output-dir reports \
    --output-prefix images_ground_truth

  uv run python Image_Processing/Utils/laplacian_confusion_matrices.py \
    --csv-path ground_truth.csv \
    --image-root Images/AURA1612 \
    --thresholds 100 180 \
    --output-dir reports \
    --output-prefix aura1612_ground_truth
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {".git", ".venv", ".uv-cache", ".uv_cache", ".cache", ".mplconfig", "__pycache__"}

LABEL_POSITIVE = "too_blurry"
LABEL_NEGATIVE = "focused_enough"


@dataclass(frozen=True)
class CoverageSummary:
    total_images: int
    unique_filenames: int
    image_duplicate_filenames: int
    images_missing_csv_row: int
    images_with_label: int
    images_with_missing_label: int
    images_with_unknown_label: int


@dataclass(frozen=True)
class RowStats:
    rows_total: int
    rows_with_empty_filename: int
    rows_with_missing_label: int
    rows_with_unknown_label: int
    rows_with_missing_image: int
    rows_with_unreadable_image: int
    rows_used: int
    csv_duplicate_filenames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Laplacian variance confusion matrices for blur detection.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth.csv"),
        help="CSV containing filenames and blur labels (default: ground_truth.csv).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Folder containing images (searched recursively).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="usability considering blur",
        help="CSV column containing blur usability labels.",
    )
    parser.add_argument(
        "--filename-column",
        type=str,
        default="filename",
        help="CSV column containing image filenames.",
    )
    parser.add_argument(
        "--project-column",
        type=str,
        default=None,
        help="Optional CSV column for project subfolders under the image root.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        required=True,
        help="One or more Laplacian variance thresholds to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write metrics JSON files (default: reports).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="laplacian_metrics",
        help="Prefix for metrics filenames.",
    )
    return parser.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d not in SKIP_DIR_NAMES
        ]
        for filename in filenames:
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            yield Path(current_root) / filename


def index_images(root: Path) -> tuple[dict[str, Path], int, int]:
    lookup: dict[str, Path] = {}
    total = 0
    duplicates = 0
    for image_path in iter_image_paths(root):
        total += 1
        key = image_path.name.lower()
        if key in lookup and lookup[key] != image_path:
            duplicates += 1
            continue
        lookup[key] = image_path
    return lookup, total, duplicates


def normalize_label(raw_value: object) -> Optional[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None
    label = str(raw_value).strip().lower()
    if not label:
        return None
    if "too blurry" in label:
        return LABEL_POSITIVE
    if "focused enough" in label:
        return LABEL_NEGATIVE
    return "unknown"


def resolve_image_path(
    filename: str,
    row: pd.Series,
    image_root: Path,
    filename_lookup: dict[str, Path],
    project_column: Optional[str],
) -> Optional[Path]:
    if project_column:
        project = str(row.get(project_column, "") or "").strip()
        if project:
            candidate = image_root / project / filename
            if candidate.exists():
                return candidate
    return filename_lookup.get(filename.lower())


def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def summarize_coverage(
    *,
    image_root: Path,
    filename_lookup: dict[str, Path],
    image_total: int,
    image_duplicates: int,
    df: pd.DataFrame,
    filename_column: str,
    label_column: str,
) -> CoverageSummary:
    image_names = {path.name.lower() for path in filename_lookup.values()}
    csv_names = set()
    csv_with_label = set()
    csv_with_missing_label = set()
    csv_with_unknown_label = set()

    for _, row in df.iterrows():
        filename = str(row.get(filename_column, "") or "").strip()
        if not filename:
            continue
        filename_key = filename.lower()
        csv_names.add(filename_key)
        label = normalize_label(row.get(label_column))
        if label is None:
            csv_with_missing_label.add(filename_key)
        elif label == "unknown":
            csv_with_unknown_label.add(filename_key)
        else:
            csv_with_label.add(filename_key)

    images_missing_csv = image_names - csv_names
    images_with_label = image_names & csv_with_label
    images_missing_label = image_names & csv_with_missing_label
    images_unknown_label = image_names & csv_with_unknown_label

    return CoverageSummary(
        total_images=image_total,
        unique_filenames=len(image_names),
        image_duplicate_filenames=image_duplicates,
        images_missing_csv_row=len(images_missing_csv),
        images_with_label=len(images_with_label),
        images_with_missing_label=len(images_missing_label),
        images_with_unknown_label=len(images_unknown_label),
    )


def compute_confusion_metrics(
    records: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    if not records:
        return {
            "threshold": threshold,
            "sample_count": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "confusion_matrix": None,
        }

    y_true = np.array([1 if r["label"] == LABEL_POSITIVE else 0 for r in records])
    y_pred = np.array([1 if r["laplacian_variance"] < threshold else 0 for r in records])

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "threshold": threshold,
        "sample_count": int(len(records)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": {
            "labels": [LABEL_POSITIVE, LABEL_NEGATIVE],
            "matrix": [[tp, fn], [fp, tn]],
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        },
    }


def format_threshold_tag(threshold: float) -> str:
    text = f"{threshold:g}"
    return text.replace(".", "p")


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser()
    image_root = args.image_root.expanduser()
    output_dir = args.output_dir.expanduser()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    df = pd.read_csv(csv_path)
    if args.filename_column not in df.columns:
        raise ValueError(f"CSV missing filename column: {args.filename_column}")
    if args.label_column not in df.columns:
        raise ValueError(f"CSV missing label column: {args.label_column}")
    if args.project_column and args.project_column not in df.columns:
        raise ValueError(f"CSV missing project column: {args.project_column}")

    filename_lookup, image_total, image_duplicates = index_images(image_root)

    coverage = summarize_coverage(
        image_root=image_root,
        filename_lookup=filename_lookup,
        image_total=image_total,
        image_duplicates=image_duplicates,
        df=df,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )

    rows_total = int(len(df))
    rows_with_empty_filename = 0
    rows_with_missing_label = 0
    rows_with_unknown_label = 0
    rows_with_missing_image = 0
    rows_with_unreadable_image = 0
    csv_duplicate_filenames = int(df[args.filename_column].astype(str).str.lower().duplicated().sum())

    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        filename = str(row.get(args.filename_column, "") or "").strip()
        if not filename:
            rows_with_empty_filename += 1
            continue

        label = normalize_label(row.get(args.label_column))
        if label is None:
            rows_with_missing_label += 1
            continue
        if label == "unknown":
            rows_with_unknown_label += 1
            continue

        image_path = resolve_image_path(
            filename,
            row,
            image_root,
            filename_lookup,
            args.project_column,
        )
        if image_path is None or not image_path.exists():
            rows_with_missing_image += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows_with_unreadable_image += 1
            continue

        variance = compute_laplacian_variance(image)
        records.append(
            {
                "filename": filename,
                "image_path": str(image_path),
                "label": label,
                "laplacian_variance": variance,
            }
        )

    row_stats = RowStats(
        rows_total=rows_total,
        rows_with_empty_filename=rows_with_empty_filename,
        rows_with_missing_label=rows_with_missing_label,
        rows_with_unknown_label=rows_with_unknown_label,
        rows_with_missing_image=rows_with_missing_image,
        rows_with_unreadable_image=rows_with_unreadable_image,
        rows_used=len(records),
        csv_duplicate_filenames=csv_duplicate_filenames,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for threshold in args.thresholds:
        metrics = compute_confusion_metrics(records, threshold=threshold)
        payload = {
            "threshold": metrics["threshold"],
            "positive_label": LABEL_POSITIVE,
            "negative_label": LABEL_NEGATIVE,
            "score_definition": "blurry if laplacian_variance < threshold",
            "sample_count": metrics["sample_count"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "counts": row_stats.__dict__,
            "image_coverage": coverage.__dict__,
            "inputs": {
                "csv_path": str(csv_path),
                "image_root": str(image_root),
                "label_column": args.label_column,
                "filename_column": args.filename_column,
                "project_column": args.project_column,
            },
        }

        tag = format_threshold_tag(float(threshold))
        output_path = output_dir / f"{args.output_prefix}_threshold_{tag}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved metrics: {output_path}")

    print("\nCoverage summary:")
    print(json.dumps(coverage.__dict__, indent=2))
    print("\nRow stats:")
    print(json.dumps(row_stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
