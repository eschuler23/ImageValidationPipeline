#!/usr/bin/env python3
"""
Pipeline component: SAM foreground mask + Laplacian variance.

How to run (from repo root):
  uv run python src/Pipeline_component_blur_gate_sam_laplacian.py \
    --image-dir AURA1612_has_ground_truth \
    --csv-path ground_truth.csv \
    --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
    --sam-model-type vit_b \
    --output-dir reports

VSCode: open this file and run it with the same arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_PREDICTIONS_NAME = "sam_foreground_laplacian_predictions.csv"
DEFAULT_METRICS_NAME = "sam_foreground_laplacian_metrics.json"


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


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_sam_dependencies():
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as exc:  # pragma: no cover - guidance for missing deps
        raise RuntimeError(
            "Missing SAM dependencies. Install torch + segment-anything first, "
            "then re-run this script."
        ) from exc
    return torch, SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def resolve_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resize_for_sam(image_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return image_rgb, 1.0
    h, w = image_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return image_rgb, 1.0
    scale = max_side / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def build_prompt_points(image_shape: Tuple[int, int, int], corner_offset_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    offset = max(1, int(round(min(h, w) * corner_offset_ratio)))
    center_x = float(w - 1) / 2.0
    center_y = float(h - 1) / 2.0

    points = [
        [center_x, center_y],
        [float(max(0, min(w - 1, offset))), float(max(0, min(h - 1, offset)))],
        [float(max(0, min(w - 1, w - 1 - offset))), float(max(0, min(h - 1, offset)))],
        [float(max(0, min(w - 1, offset))), float(max(0, min(h - 1, h - 1 - offset)))],
        [float(max(0, min(w - 1, w - 1 - offset))), float(max(0, min(h - 1, h - 1 - offset)))],
    ]
    labels = [1, 0, 0, 0, 0]
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_center_box_mask(height: int, width: int, center_ratio: float) -> np.ndarray:
    center_ratio = float(center_ratio)
    if center_ratio <= 0 or center_ratio > 1:
        raise ValueError("center_ratio must be in (0, 1].")
    box_w = max(1, int(round(width * center_ratio)))
    box_h = max(1, int(round(height * center_ratio)))
    x0 = max(0, (width - box_w) // 2)
    y0 = max(0, (height - box_h) // 2)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + box_h, x0 : x0 + box_w] = True
    return mask


def build_border_mask(height: int, width: int, border_ratio: float) -> np.ndarray:
    border_ratio = float(border_ratio)
    if border_ratio <= 0:
        return np.zeros((height, width), dtype=bool)
    border_width = max(1, int(round(min(height, width) * border_ratio)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True
    return mask


def score_mask_geometry(mask: np.ndarray, center_mask: np.ndarray, border_mask: np.ndarray) -> Tuple[float, float, float]:
    area = float(mask.sum())
    if area <= 0:
        return 0.0, 0.0, 0.0
    center_ratio = float(mask[center_mask].sum() / area)
    border_ratio = float(mask[border_mask].sum() / area)
    return area, center_ratio, border_ratio


def select_auto_mask(
    masks: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    center_ratio: float,
    border_ratio: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Optional[float]]]:
    if not masks:
        return None, {"mask_score": None, "mask_center_ratio": None, "mask_border_ratio": None}

    h, w = image_shape[:2]
    image_area = float(h * w)
    center_mask = build_center_box_mask(h, w, center_ratio)
    border_mask = build_border_mask(h, w, border_ratio)

    best = None
    best_score = -1.0
    best_meta: Dict[str, Optional[float]] = {"mask_score": None, "mask_center_ratio": None, "mask_border_ratio": None}

    for item in masks:
        segmentation = item.get("segmentation")
        if segmentation is None:
            continue
        mask = segmentation.astype(bool)
        area, center_overlap, border_overlap = score_mask_geometry(mask, center_mask, border_mask)
        area_ratio = area / image_area if image_area > 0 else 0.0
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        predicted_iou = float(item.get("predicted_iou", 0.0))
        score = predicted_iou + center_overlap - border_overlap
        if score > best_score:
            best = mask
            best_score = score
            best_meta = {
                "mask_score": predicted_iou,
                "mask_center_ratio": center_overlap,
                "mask_border_ratio": border_overlap,
            }

    if best is None:
        # Fall back to the highest predicted IoU if no mask meets area constraints.
        fallback = max(masks, key=lambda item: float(item.get("predicted_iou", 0.0)))
        mask = fallback.get("segmentation")
        if mask is None:
            return None, {"mask_score": None, "mask_center_ratio": None, "mask_border_ratio": None}
        mask = mask.astype(bool)
        _, center_overlap, border_overlap = score_mask_geometry(mask, center_mask, border_mask)
        return mask, {
            "mask_score": float(fallback.get("predicted_iou", 0.0)),
            "mask_center_ratio": center_overlap,
            "mask_border_ratio": border_overlap,
        }

    return best, best_meta


def generate_prompt_mask(predictor, image_rgb: np.ndarray, corner_offset_ratio: float) -> Tuple[Optional[np.ndarray], Dict[str, Optional[float]]]:
    predictor.set_image(image_rgb)
    points, labels = build_prompt_points(image_rgb.shape, corner_offset_ratio)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        return None, {"mask_score": None, "mask_center_ratio": None, "mask_border_ratio": None}
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), {
        "mask_score": float(scores[best_idx]),
        "mask_center_ratio": None,
        "mask_border_ratio": None,
    }


def compute_foreground_laplacian_variance(
    image_bgr: np.ndarray, mask: np.ndarray, erode_iterations: int
) -> Tuple[Optional[float], int, np.ndarray]:
    mask_uint8 = mask.astype(np.uint8)
    if erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=erode_iterations)
    mask_bool = mask_uint8.astype(bool)
    if not mask_bool.any():
        return None, 0, mask_uint8

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    values = laplacian[mask_bool]
    return float(np.var(values)), int(values.size), mask_uint8


def summarize_mask_ratios(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def find_best_threshold(variances: np.ndarray, labels: np.ndarray) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    if variances.size == 0 or len(np.unique(labels)) < 2:
        return None, {"best_f1": None, "best_precision": None, "best_recall": None, "candidate_count": 0}

    quantiles = np.linspace(0.05, 0.95, 19)
    candidates = np.unique(np.quantile(variances, quantiles))
    best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for threshold in candidates:
        y_pred = (variances < threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary", pos_label=1, zero_division=0
        )
        f1 = f1_score(labels, y_pred, average="weighted", zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best["threshold"], {
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "candidate_count": int(len(candidates)),
    }


def apply_threshold(records: List[Dict[str, Any]], threshold: Optional[float]) -> None:
    for row in records:
        if row.get("status") != "ok" or row.get("laplacian_variance") is None:
            row["prediction_label"] = None
            row["prediction_binary"] = None
            continue
        if threshold is None:
            row["prediction_label"] = None
            row["prediction_binary"] = None
            continue
        predicted_blurry = row["laplacian_variance"] < threshold
        row["prediction_label"] = "blurry" if predicted_blurry else "not_blurry"
        row["prediction_binary"] = 1 if predicted_blurry else 0


def compute_metrics(records: List[Dict[str, Any]], threshold: Optional[float]) -> Dict[str, Any]:
    labeled = [
        row
        for row in records
        if row.get("status") == "ok" and row.get("ground_truth_binary") in (0, 1)
    ]

    metrics: Dict[str, Any] = {
        "threshold": threshold,
        "positive_label": "too_blurry",
        "negative_label": "focused_enough",
        "score_definition": "blur_score = 1 / (laplacian_variance + 1e-12)",
        "sample_count": int(len(labeled)),
    }

    if labeled:
        y_true = np.array([row["ground_truth_binary"] for row in labeled], dtype=int)
        y_scores = np.array([row["blur_score"] for row in labeled], dtype=float)

        if threshold is not None:
            y_pred = np.array([row["prediction_binary"] for row in labeled], dtype=int)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)
            metrics["f1"] = float(f1)
            metrics["accuracy"] = float(np.mean(y_true == y_pred))

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
        else:
            metrics["precision"] = None
            metrics["recall"] = None
            metrics["f1"] = None
            metrics["accuracy"] = None
            metrics["confusion_matrix"] = None

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
        description="Blur gate: SAM foreground mask + Laplacian variance.",
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
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="Path to the SAM checkpoint file (local).",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type matching the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for SAM.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="prompt",
        choices=["prompt", "auto"],
        help="Mask strategy: prompt uses center+corner points; auto uses SAM auto masks.",
    )
    parser.add_argument(
        "--sam-max-side",
        type=int,
        default=0,
        help="Optional max side for SAM input (resize for speed).",
    )
    parser.add_argument(
        "--corner-offset-ratio",
        type=float,
        default=0.05,
        help="Offset ratio for negative corner prompts (prompt mode).",
    )
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.02,
        help="Minimum foreground mask area as ratio of image size.",
    )
    parser.add_argument(
        "--max-mask-area-ratio",
        type=float,
        default=0.9,
        help="Maximum foreground mask area as ratio of image size.",
    )
    parser.add_argument(
        "--center-box-ratio",
        type=float,
        default=0.5,
        help="Center box ratio for auto mask scoring.",
    )
    parser.add_argument(
        "--border-width-ratio",
        type=float,
        default=0.02,
        help="Border width ratio used to penalize background-heavy masks.",
    )
    parser.add_argument(
        "--mask-erode-iterations",
        type=int,
        default=2,
        help="Erode the mask before computing Laplacian variance to reduce edge artifacts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional Laplacian variance threshold; if omitted we choose the best F1 on labeled data.",
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
    parser.add_argument(
        "--auto-points-per-side",
        type=int,
        default=32,
        help="SAM auto mask generator points per side.",
    )
    parser.add_argument(
        "--auto-pred-iou-thresh",
        type=float,
        default=0.86,
        help="SAM auto mask generator predicted IoU threshold.",
    )
    parser.add_argument(
        "--auto-stability-thresh",
        type=float,
        default=0.92,
        help="SAM auto mask generator stability threshold.",
    )
    parser.add_argument(
        "--auto-min-region-area",
        type=int,
        default=100,
        help="SAM auto mask generator minimum mask region area (pixels).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    torch_module, SamAutomaticMaskGenerator, SamPredictor, sam_model_registry = load_sam_dependencies()
    device = resolve_device(torch_module, args.device)

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    mask_generator = None
    if args.mask_mode == "auto":
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=args.auto_points_per_side,
            pred_iou_thresh=args.auto_pred_iou_thresh,
            stability_score_thresh=args.auto_stability_thresh,
            min_mask_region_area=args.auto_min_region_area,
        )

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

    inference_context = torch_module.inference_mode if hasattr(torch_module, "inference_mode") else torch_module.no_grad

    rows = df.to_dict(orient="records")
    with inference_context():
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
                        "mask_mode": args.mask_mode,
                        "mask_status": "missing_image",
                        "mask_score": None,
                        "mask_area_ratio": None,
                        "mask_eroded_area_ratio": None,
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
                        "mask_mode": args.mask_mode,
                        "mask_status": "unreadable_image",
                        "mask_score": None,
                        "mask_area_ratio": None,
                        "mask_eroded_area_ratio": None,
                        "laplacian_variance": None,
                        "blur_score": None,
                        "prediction_label": None,
                        "prediction_binary": None,
                    }
                )
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_image, scale = resize_for_sam(image_rgb, args.sam_max_side)

            mask: Optional[np.ndarray] = None
            mask_meta: Dict[str, Optional[float]] = {
                "mask_score": None,
                "mask_center_ratio": None,
                "mask_border_ratio": None,
            }

            if args.mask_mode == "prompt":
                mask, mask_meta = generate_prompt_mask(predictor, sam_image, args.corner_offset_ratio)
            else:
                auto_masks = mask_generator.generate(sam_image) if mask_generator is not None else []
                mask, mask_meta = select_auto_mask(
                    auto_masks,
                    sam_image.shape,
                    min_area_ratio=args.min_mask_area_ratio,
                    max_area_ratio=args.max_mask_area_ratio,
                    center_ratio=args.center_box_ratio,
                    border_ratio=args.border_width_ratio,
                )

            if mask is not None and scale != 1.0:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)

            if mask is None:
                records.append(
                    {
                        "filename": filename,
                        "image_path": rel_image_path,
                        "status": "mask_failed",
                        "ground_truth_label_raw": label_raw_clean,
                        "ground_truth_label": label_norm,
                        "ground_truth_binary": label_binary,
                        "mask_mode": args.mask_mode,
                        "mask_status": "no_mask",
                        "mask_score": mask_meta.get("mask_score"),
                        "mask_area_ratio": None,
                        "mask_eroded_area_ratio": None,
                        "laplacian_variance": None,
                        "blur_score": None,
                        "prediction_label": None,
                        "prediction_binary": None,
                    }
                )
                continue

            mask_area = int(mask.sum())
            image_area = int(image.shape[0] * image.shape[1])
            mask_area_ratio = float(mask_area) / float(image_area) if image_area > 0 else 0.0

            mask_status = "ok"
            if mask_area_ratio < args.min_mask_area_ratio:
                mask_status = "mask_too_small"
            elif mask_area_ratio > args.max_mask_area_ratio:
                mask_status = "mask_too_large"

            variance: Optional[float] = None
            blur_score: Optional[float] = None
            eroded_ratio: Optional[float] = None

            if mask_status == "ok":
                variance, pixel_count, eroded_mask = compute_foreground_laplacian_variance(
                    image, mask, args.mask_erode_iterations
                )
                eroded_ratio = float(eroded_mask.sum()) / float(image_area) if image_area > 0 else None
                if variance is None:
                    mask_status = "mask_eroded_empty"

            if mask_status == "ok" and variance is not None:
                blur_score = 1.0 / (variance + 1e-12)

            status = "ok" if mask_status == "ok" else "mask_failed"
            records.append(
                {
                    "filename": filename,
                    "image_path": rel_image_path,
                    "status": status,
                    "ground_truth_label_raw": label_raw_clean,
                    "ground_truth_label": label_norm,
                    "ground_truth_binary": label_binary,
                    "mask_mode": args.mask_mode,
                    "mask_status": mask_status,
                    "mask_score": mask_meta.get("mask_score"),
                    "mask_area_ratio": mask_area_ratio,
                    "mask_eroded_area_ratio": eroded_ratio,
                    "mask_center_ratio": mask_meta.get("mask_center_ratio"),
                    "mask_border_ratio": mask_meta.get("mask_border_ratio"),
                    "laplacian_variance": variance,
                    "blur_score": blur_score,
                    "prediction_label": None,
                    "prediction_binary": None,
                }
            )

    labeled_variances = np.array(
        [
            row["laplacian_variance"]
            for row in records
            if row.get("status") == "ok" and row.get("ground_truth_binary") in (0, 1)
        ],
        dtype=float,
    )
    labeled_labels = np.array(
        [
            row["ground_truth_binary"]
            for row in records
            if row.get("status") == "ok" and row.get("ground_truth_binary") in (0, 1)
        ],
        dtype=int,
    )

    threshold = args.threshold
    threshold_source = "user"
    threshold_search = None
    if threshold is None:
        threshold, threshold_search = find_best_threshold(labeled_variances, labeled_labels)
        threshold_source = "best_f1"

    apply_threshold(records, threshold)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / args.predictions_name
    metrics_path = output_dir / args.metrics_name

    predictions_df = pd.DataFrame.from_records(records)
    predictions_df.to_csv(predictions_path, index=False)

    metrics = compute_metrics(records, threshold)
    metrics["threshold_source"] = threshold_source
    metrics["threshold_search"] = threshold_search
    metrics["counts"] = {
        "rows_total": int(len(df)),
        "rows_with_empty_filename": int(empty_filename_rows),
        "records_written": int(len(records)),
        "missing_images": int(missing_images),
        "unreadable_images": int(unreadable_images),
        "mask_failed": int(sum(1 for row in records if row.get("status") == "mask_failed")),
    }

    mask_ratios = [
        float(row["mask_area_ratio"])
        for row in records
        if row.get("status") == "ok" and row.get("mask_area_ratio") is not None
    ]
    metrics["mask_stats"] = {
        "area_ratio": summarize_mask_ratios(mask_ratios),
        "erode_iterations": int(args.mask_erode_iterations),
        "min_area_ratio": float(args.min_mask_area_ratio),
        "max_area_ratio": float(args.max_mask_area_ratio),
    }
    metrics["inputs"] = {
        "image_dir": str(args.image_dir),
        "csv_path": str(args.csv_path),
        "label_column": args.label_column,
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_model_type": args.sam_model_type,
        "mask_mode": args.mask_mode,
        "device": device,
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
