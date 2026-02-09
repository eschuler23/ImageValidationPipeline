#!/usr/bin/env python3
"""
Create ROI review images for manual inspection.

This mirrors the archive-style visual review but focuses on ROI cropping:
- draws the ROI box on the original image
- shows the cropped ROI next to it
- labels each tile with ground truth, prediction, and score

How to run (from repo root):
  uv run python src/Pipeline_component_blur_gate_roi_review_images.py \
    --image-dir AURA1612_has_ground_truth \
    --predictions-csv reports/roi_patch_predictions.csv \
    --output-dir "specs/Blurr_detection - SAM foreground/artifacts/roi_review" \
    --subset errors
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


LABEL_POSITIVE = "too_blurry"
LABEL_NEGATIVE = "focused_enough"


def label_to_gt(label: str) -> Optional[str]:
    label = (label or "").strip().lower()
    if label == LABEL_POSITIVE:
        return "blurry"
    if label == LABEL_NEGATIVE:
        return "not_blurry"
    return None


def center_crop_box(width: int, height: int, ratio: float) -> Tuple[int, int, int, int]:
    if ratio <= 0 or ratio > 1:
        raise ValueError("center crop ratio must be in (0, 1].")
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    x1 = min(width, x0 + crop_w)
    y1 = min(height, y0 + crop_h)
    return x0, y0, x1, y1


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if height <= 0:
        return image
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)


def draw_label(
    image: np.ndarray, text: str, *, color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = margin
    y = margin + text_h
    box_x1 = min(out.shape[1] - 1, x + text_w + 8)
    box_y1 = min(out.shape[0] - 1, y + baseline + 8)

    # Dark background to keep text readable on bright images.
    cv2.rectangle(out, (x - 4, y - text_h - 4), (box_x1, box_y1), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return out


def build_grid(images: List[np.ndarray], cols: int, tile_size: Tuple[int, int]) -> np.ndarray:
    if not images:
        raise ValueError("No images supplied to grid.")
    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / cols))
    tile_w, tile_h = tile_size

    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0 = r * tile_h
        x0 = c * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = resized
    return grid


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ROI review images.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subset",
        type=str,
        default="errors",
        choices=["errors", "false_positives", "false_negatives", "all"],
        help="Which subset to render.",
    )
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=None,
        help="Fallback crop ratio if predictions CSV does not include it.",
    )
    parser.add_argument("--tile-height", type=int, default=320)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument(
        "--grid-max-images",
        type=int,
        default=None,
        help="Optional max images per grid file; split into multiple grids when set.",
    )
    parser.add_argument("--max-per-group", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {args.predictions_csv}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {args.image_dir}")

    df = pd.read_csv(args.predictions_csv)
    required = {"filename", "ground_truth_label", "predicted_label", "predicted_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df["gt_label"] = df["ground_truth_label"].map(label_to_gt)
    df = df[df["gt_label"].notna() & df["predicted_label"].notna()].copy()

    df["error_type"] = "correct"
    df.loc[
        (df["gt_label"] == "not_blurry") & (df["predicted_label"] == "blurry"),
        "error_type",
    ] = "false_positive"
    df.loc[
        (df["gt_label"] == "blurry") & (df["predicted_label"] == "not_blurry"),
        "error_type",
    ] = "false_negative"

    if args.subset == "false_positives":
        subset_df = df[df["error_type"] == "false_positive"].copy()
    elif args.subset == "false_negatives":
        subset_df = df[df["error_type"] == "false_negative"].copy()
    elif args.subset == "errors":
        subset_df = df[df["error_type"].isin(["false_positive", "false_negative"])].copy()
    else:
        subset_df = df.copy()

    if subset_df.empty:
        print("No images to render for subset:", args.subset)
        return 0

    output_dir = args.output_dir
    ensure_dir(output_dir)

    grouped: Dict[str, pd.DataFrame] = {}
    for name, group in subset_df.groupby("error_type"):
        grouped[name] = group.sort_values("predicted_score", ascending=False)

    for error_type, group in grouped.items():
        images: List[np.ndarray] = []
        group_dir = output_dir / error_type
        ensure_dir(group_dir)

        ascending = error_type == "false_negative"
        group = group.sort_values("predicted_score", ascending=ascending)

        for _, row in group.iterrows():
            if args.max_per_group is not None and len(images) >= args.max_per_group:
                break
            filename = str(row["filename"])
            image_path = args.image_dir / filename
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            ratio = row.get("center_crop_ratio")
            if pd.isna(ratio):
                ratio = args.center_crop_ratio
            if ratio is None:
                raise ValueError("center crop ratio not found in CSV and not provided.")

            ratio = float(ratio)
            h, w = image.shape[:2]
            x0, y0, x1, y1 = center_crop_box(w, h, ratio)

            annotated = image.copy()
            color = (0, 0, 255) if error_type == "false_positive" else (255, 0, 0)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 3)

            roi = image[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            label_text = (
                f"{row['filename']} | GT={row['gt_label']} "
                f"| Pred={row['predicted_label']} | score={row['predicted_score']:.4f}"
            )
            annotated = draw_label(annotated, label_text)

            annotated = resize_to_height(annotated, args.tile_height)
            roi = resize_to_height(roi, args.tile_height)

            combined = cv2.hconcat([annotated, roi])
            images.append(combined)

            out_path = group_dir / filename
            cv2.imwrite(str(out_path), combined)

        if not images:
            continue

        grid_max = args.grid_max_images
        if grid_max is None or grid_max <= 0:
            grid_max = len(images)

        for idx in range(0, len(images), grid_max):
            chunk = images[idx : idx + grid_max]
            grid = build_grid(chunk, args.grid_cols, (args.tile_width, args.tile_height))
            if idx == 0:
                grid_name = f"{error_type}_grid.png"
            else:
                grid_name = f"{error_type}_grid_{(idx // grid_max) + 1:02d}.png"
            grid_path = output_dir / grid_name
            cv2.imwrite(str(grid_path), grid)
            print("Wrote grid:", grid_path)

    print("Saved review images to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
