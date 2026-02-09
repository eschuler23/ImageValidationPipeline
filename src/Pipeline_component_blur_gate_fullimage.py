#!/usr/bin/env python3
"""
Pipeline component: Blur gate baseline (full-image Laplacian variance).

How to run (from repo root):
  uv run python src/Pipeline_component_blur_gate_fullimage.py \
    --image-dir AURA1612_has_ground_truth \
    --csv-path ground_truth.csv \
    --output-dir reports \
    --threshold 186.895

VSCode: open this file and run it with the same arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DEFAULT_IMAGE_DIR = Path("AURA1612_has_ground_truth")
DEFAULT_CSV_PATH = Path("ground_truth.csv")
DEFAULT_LABEL_COLUMN = "usability considering blur"
DEFAULT_THRESHOLD = 186.895  # Matches the archive baseline; adjust if needed.

DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_PREDICTIONS_NAME = "baseline_fullimage_predictions.csv"
DEFAULT_METRICS_NAME = "baseline_fullimage_metrics.json"


def normalize_usability_label(raw_value: Any) -> str:
    """Normalize the CSV usability label into a stable class name."""
    if raw_value is None or pd.isna(raw_value):
        return "unknown"
    label = str(raw_value).strip().lower()
    if not label:
        return "unknown"
    if "too blurry" in label:
        return "too_blurry"
    if "focused enough" in label:
        return "focused_enough"
    return "unknown"


def label_to_binary(label: str) -> Optional[int]:
    """Map normalized labels to binary, where 1 = too_blurry (positive class)."""
    if label == "too_blurry":
        return 1
    if label == "focused_enough":
        return 0
    return None


def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    """Compute Laplacian variance (higher means sharper)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_relpath(path: Path, base: Path) -> str:
    """Best-effort relative path for nicer CSV output."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def compute_metrics(
    records: List[Dict[str, Any]],
    *,
    threshold: float,
    score_field: str = "blur_score",
) -> Dict[str, Any]:
    labeled = [
        row
        for row in records
        if row.get("status") == "ok" and row.get("ground_truth_binary") in (0, 1)
    ]

    y_true = np.array([row["ground_truth_binary"] for row in labeled], dtype=int)
    y_pred = np.array([row["prediction_binary"] for row in labeled], dtype=int)
    y_scores = np.array([row[score_field] for row in labeled], dtype=float)

    metrics: Dict[str, Any] = {
        "threshold": threshold,
        "positive_label": "too_blurry",
        "negative_label": "focused_enough",
        "score_definition": "blur_score = 1 / (laplacian_variance + 1e-12)",
        "sample_count": int(len(labeled)),
    }

    if labeled:
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        accuracy = float(np.mean(y_true == y_pred))
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)
        metrics["accuracy"] = accuracy

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tp, fn, fp, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        metrics["confusion_matrix"] = {
            "labels": ["too_blurry", "focused_enough"],
            "matrix": cm.tolist(),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }

        if len(np.unique(y_true)) < 2:
            metrics["pr_auc"] = None
            metrics["pr_auc_note"] = "Only one class present; PR-AUC is undefined."
        else:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_scores))
    else:
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1"] = None
        metrics["accuracy"] = None
        metrics["confusion_matrix"] = None
        metrics["pr_auc"] = None
        metrics["pr_auc_note"] = "No labeled samples available."

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline blur gate: full-image Laplacian variance.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing images listed in the CSV.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV with ground truth labels.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="CSV column name for blur usability labels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Laplacian variance threshold; lower is sharper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write reports.",
    )
    parser.add_argument(
        "--predictions-name",
        type=str,
        default=DEFAULT_PREDICTIONS_NAME,
        help="Filename for the predictions CSV.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default=DEFAULT_METRICS_NAME,
        help="Filename for the metrics JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick debugging.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    df = pd.read_csv(args.csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV must include a 'filename' column.")
    if args.label_column not in df.columns:
        raise ValueError(f"CSV missing label column: {args.label_column!r}")

    if args.limit is not None:
        df = df.head(int(args.limit))

    records: List[Dict[str, Any]] = []
    missing_images = 0
    unreadable_images = 0
    empty_filename_rows = 0

    rows = df.to_dict(orient="records")
    for row in tqdm(rows, desc="Scoring images", unit="image"):
        filename = str(row.get("filename", "")).strip()
        if not filename:
            empty_filename_rows += 1
            continue

        label_raw = row.get(args.label_column)
        label_raw_clean = None if pd.isna(label_raw) else label_raw
        label_norm = normalize_usability_label(label_raw_clean)
        label_binary = label_to_binary(label_norm)

        image_path = args.image_dir / filename
        rel_image_path = safe_relpath(image_path, Path.cwd())

        if not image_path.exists():
            missing_images += 1
            records.append(
                {
                    "filename": filename,
                    "image_path": rel_image_path,
                    "status": "missing_image",
                    "ground_truth_label_raw": label_raw_clean,
                    "ground_truth_label": label_norm,
                    "ground_truth_binary": label_binary,
                    "laplacian_variance": None,
                    "blur_score": None,
                    "prediction_label": None,
                    "prediction_binary": None,
                }
            )
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_images += 1
            records.append(
                {
                    "filename": filename,
                    "image_path": rel_image_path,
                    "status": "unreadable_image",
                    "ground_truth_label_raw": label_raw_clean,
                    "ground_truth_label": label_norm,
                    "ground_truth_binary": label_binary,
                    "laplacian_variance": None,
                    "blur_score": None,
                    "prediction_label": None,
                    "prediction_binary": None,
                }
            )
            continue

        variance = compute_laplacian_variance(image)
        # Invert variance so higher score means "more blurry" for PR-AUC.
        blur_score = 1.0 / (variance + 1e-12)
        predicted_blurry = variance < args.threshold
        prediction_label = "blurry" if predicted_blurry else "not_blurry"
        prediction_binary = 1 if predicted_blurry else 0

        records.append(
            {
                "filename": filename,
                "image_path": rel_image_path,
                "status": "ok",
                "ground_truth_label_raw": label_raw_clean,
                "ground_truth_label": label_norm,
                "ground_truth_binary": label_binary,
                "laplacian_variance": variance,
                "blur_score": blur_score,
                "prediction_label": prediction_label,
                "prediction_binary": prediction_binary,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / args.predictions_name
    metrics_path = output_dir / args.metrics_name

    predictions_df = pd.DataFrame.from_records(records)
    predictions_df.to_csv(predictions_path, index=False)

    metrics = compute_metrics(records, threshold=args.threshold)
    metrics["counts"] = {
        "rows_total": int(len(df)),
        "rows_with_empty_filename": int(empty_filename_rows),
        "records_written": int(len(records)),
        "missing_images": int(missing_images),
        "unreadable_images": int(unreadable_images),
    }
    metrics["inputs"] = {
        "image_dir": str(args.image_dir),
        "csv_path": str(args.csv_path),
        "label_column": args.label_column,
    }
    metrics["outputs"] = {
        "predictions_csv": str(predictions_path),
        "metrics_json": str(metrics_path),
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Wrote predictions to:", predictions_path)
    print("Wrote metrics to:", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
