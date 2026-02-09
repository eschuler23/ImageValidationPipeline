#!/usr/bin/env python3
"""
Pipeline component: Blur gate with ROI + patch-based focus metrics (Phase 2).

How to run (from repo root):
  uv run python src/Pipeline_component_blur_gate_roi_patch.py \
    --image-dir AURA1612_has_ground_truth \
    --csv-path ground_truth.csv \
    --output-dir reports \
    --center-crop-ratio 0.6 \
    --patch-grid 3 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DEFAULT_IMAGE_DIR = Path("AURA1612_has_ground_truth")
DEFAULT_CSV_PATH = Path("ground_truth.csv")
DEFAULT_LABEL_COLUMN = "usability considering blur"

DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_FEATURES_NAME = "roi_patch_features.csv"
DEFAULT_PREDICTIONS_NAME = "roi_patch_predictions.csv"
DEFAULT_METRICS_NAME = "roi_patch_metrics.json"


def normalize_usability_label(raw_value: Any) -> str:
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
    if label == "too_blurry":
        return 1
    if label == "focused_enough":
        return 0
    return None


def resize_if_needed(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0 or ratio > 1:
        raise ValueError("center_crop ratio must be in (0, 1].")
    h, w = image.shape[:2]
    crop_w = max(1, int(round(w * ratio)))
    crop_h = max(1, int(round(h * ratio)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def compute_laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_tenengrad(gray: np.ndarray) -> float:
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude_sq = grad_x ** 2 + grad_y ** 2
    return float(np.mean(magnitude_sq))


def compute_fft_highfreq_ratio(gray: np.ndarray, low_freq_frac: float = 0.1, max_size: int = 128) -> float:
    if max_size > 0:
        h, w = gray.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32, copy=False)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift) ** 2

    h, w = mag.shape[:2]
    low_h = max(1, int(round(h * low_freq_frac)))
    low_w = max(1, int(round(w * low_freq_frac)))
    cy, cx = h // 2, w // 2

    y0 = max(0, cy - low_h // 2)
    y1 = min(h, y0 + low_h)
    x0 = max(0, cx - low_w // 2)
    x1 = min(w, x0 + low_w)

    low_energy = float(np.sum(mag[y0:y1, x0:x1]))
    total_energy = float(np.sum(mag))
    if total_energy <= 0:
        return 0.0
    return float((total_energy - low_energy) / total_energy)


def build_patch_boxes(height: int, width: int, rows: int, cols: int) -> List[Tuple[int, int, int, int]]:
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)

    boxes = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if y1 > y0 and x1 > x0:
                boxes.append((y0, y1, x0, x1))
    return boxes


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return {"median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    arr = arr[np.isfinite(arr)]
    return {
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


@dataclass
class FeatureConfig:
    center_crop_ratio: float
    patch_rows: int
    patch_cols: int
    min_patch_side: int
    fft_low_freq_frac: float
    fft_max_size: int


def compute_features(image_bgr: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
    roi = center_crop(image_bgr, config.center_crop_ratio)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    features: Dict[str, float] = {
        "roi_laplacian": compute_laplacian_variance(roi_gray),
        "roi_tenengrad": compute_tenengrad(roi_gray),
        "roi_fft_ratio": compute_fft_highfreq_ratio(
            roi_gray, low_freq_frac=config.fft_low_freq_frac, max_size=config.fft_max_size
        ),
    }

    patch_boxes = build_patch_boxes(roi.shape[0], roi.shape[1], config.patch_rows, config.patch_cols)
    laplacian_scores: List[float] = []
    tenengrad_scores: List[float] = []
    fft_scores: List[float] = []

    for y0, y1, x0, x1 in patch_boxes:
        patch = roi[y0:y1, x0:x1]
        if patch.shape[0] < config.min_patch_side or patch.shape[1] < config.min_patch_side:
            continue
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        laplacian_scores.append(compute_laplacian_variance(patch_gray))
        tenengrad_scores.append(compute_tenengrad(patch_gray))
        fft_scores.append(
            compute_fft_highfreq_ratio(
                patch_gray, low_freq_frac=config.fft_low_freq_frac, max_size=config.fft_max_size
            )
        )

    for prefix, values in [
        ("laplacian", laplacian_scores),
        ("tenengrad", tenengrad_scores),
        ("fft_ratio", fft_scores),
    ]:
        stats = summarize(values)
        for suffix, value in stats.items():
            features[f"{prefix}_{suffix}"] = value

    features["patch_count"] = float(len(laplacian_scores))
    return features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = float(np.mean(y_true == y_pred))
    metrics: Dict[str, Any] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
    }

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
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blur gate with ROI + patch-based focus metrics.",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features-name", type=str, default=DEFAULT_FEATURES_NAME)
    parser.add_argument("--predictions-name", type=str, default=DEFAULT_PREDICTIONS_NAME)
    parser.add_argument("--metrics-name", type=str, default=DEFAULT_METRICS_NAME)
    parser.add_argument("--center-crop-ratio", type=float, default=0.6)
    parser.add_argument("--patch-grid", type=int, nargs=2, default=[3, 3], metavar=("ROWS", "COLS"))
    parser.add_argument("--min-patch-side", type=int, default=32)
    parser.add_argument("--fft-low-freq-frac", type=float, default=0.1)
    parser.add_argument("--fft-max-size", type=int, default=128)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
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

    config = FeatureConfig(
        center_crop_ratio=float(args.center_crop_ratio),
        patch_rows=int(args.patch_grid[0]),
        patch_cols=int(args.patch_grid[1]),
        min_patch_side=int(args.min_patch_side),
        fft_low_freq_frac=float(args.fft_low_freq_frac),
        fft_max_size=int(args.fft_max_size),
    )

    records: List[Dict[str, Any]] = []
    missing_images = 0
    unreadable_images = 0
    empty_filename_rows = 0

    rows = df.to_dict(orient="records")
    for row in tqdm(rows, desc="Extracting ROI/patch features", unit="image"):
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
                }
            )
            continue

        image = resize_if_needed(image, args.max_side)
        features = compute_features(image, config)
        record = {
            "filename": filename,
            "image_path": rel_image_path,
            "status": "ok",
            "ground_truth_label_raw": label_raw_clean,
            "ground_truth_label": label_norm,
            "ground_truth_binary": label_binary,
            "center_crop_ratio": config.center_crop_ratio,
            "patch_rows": config.patch_rows,
            "patch_cols": config.patch_cols,
        }
        record.update(features)
        records.append(record)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / args.features_name
    predictions_path = output_dir / args.predictions_name
    metrics_path = output_dir / args.metrics_name

    features_df = pd.DataFrame.from_records(records)
    features_df.to_csv(features_path, index=False)

    metrics: Dict[str, Any] = {
        "counts": {
            "rows_total": int(len(df)),
            "rows_with_empty_filename": int(empty_filename_rows),
            "records_written": int(len(records)),
            "missing_images": int(missing_images),
            "unreadable_images": int(unreadable_images),
        },
        "inputs": {
            "image_dir": str(args.image_dir),
            "csv_path": str(args.csv_path),
            "label_column": args.label_column,
        },
        "config": {
            "center_crop_ratio": config.center_crop_ratio,
            "patch_rows": config.patch_rows,
            "patch_cols": config.patch_cols,
            "min_patch_side": config.min_patch_side,
            "fft_low_freq_frac": config.fft_low_freq_frac,
            "fft_max_size": config.fft_max_size,
            "max_side": args.max_side,
        },
        "outputs": {
            "features_csv": str(features_path),
            "predictions_csv": str(predictions_path),
            "metrics_json": str(metrics_path),
        },
    }

    if args.skip_train:
        metrics["note"] = "Training skipped; only features were generated."
        features_df.to_csv(predictions_path, index=False)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print("Wrote features to:", features_path)
        print("Training skipped; metrics saved to:", metrics_path)
        return 0

    labeled_df = features_df[
        features_df["ground_truth_binary"].isin([0, 1]) & (features_df["status"] == "ok")
    ].copy()

    feature_cols = [
        col
        for col in labeled_df.columns
        if col.endswith(("_median", "_p90", "_p95", "_max"))
        or col.startswith("roi_")
    ]

    if not feature_cols:
        raise ValueError("No feature columns found; check feature extraction.")

    X = labeled_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = labeled_df["ground_truth_binary"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        labeled_df.index.to_numpy(),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics.update(compute_metrics(y_test, y_pred, y_score))
    metrics["train_test_split"] = {
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "features": feature_cols,
    }

    # Save predictions for all labeled rows (model trained on train split).
    labeled_scores = model.predict_proba(X)[:, 1]
    labeled_predictions = (labeled_scores >= 0.5).astype(int)

    labeled_df["predicted_score"] = labeled_scores
    labeled_df["predicted_binary"] = labeled_predictions
    labeled_df["predicted_label"] = labeled_df["predicted_binary"].map(
        {1: "blurry", 0: "not_blurry"}
    )

    predictions_df = features_df.merge(
        labeled_df[["predicted_score", "predicted_binary", "predicted_label"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    predictions_df.to_csv(predictions_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Wrote features to:", features_path)
    print("Wrote predictions to:", predictions_path)
    print("Wrote metrics to:", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
