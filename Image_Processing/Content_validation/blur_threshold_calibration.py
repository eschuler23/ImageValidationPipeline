#!/usr/bin/env python3
"""
Generate blur-augmented grids to choose blur thresholds.

This tool saves a grid per sample showing original + multiple blur radii.
Use the output markdown file to review and decide:
- a blur range that keeps the original label
- a blur range that flips to "not usable"

How to run (example):
  uv run python Image_Processing/Content_validation/blur_threshold_calibration.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --project-column project \
    --filename-column filename \
    --label-column "usability considering nfp" \
    --positive-labels "usable" \
    --negative-labels "not usable" \
    --decode-percent-newlines \
    --max-images 12 \
    --blur-range 0.5 4.0 0.5 \
    --output-dir Image_Processing/Content_validation/blur_threshold_review
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


@dataclass
class Sample:
    image_path: Path
    label: int
    label_raw: str
    filename: str
    project: str


def normalize_label(label: str) -> str:
    return label.strip().lower()


def decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def parse_label_list(values: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for value in values:
        for piece in value.split(","):
            cleaned = normalize_label(piece)
            if cleaned:
                labels.append(cleaned)
    return labels


def expand_range(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) != 3:
        raise ValueError("Range values must be: min max step")
    start, end, step = (float(values[0]), float(values[1]), float(values[2]))
    if step <= 0:
        raise ValueError("Range step must be > 0")
    radii: List[float] = []
    current = start
    while current <= end + 1e-9:
        radii.append(round(current, 6))
        current += step
    return radii


def load_samples(
    csv_path: Path,
    image_root: Path,
    label_column: str,
    positive_labels: Sequence[str],
    negative_labels: Sequence[str],
    filename_column: str,
    project_column: str,
    decode_newlines: bool,
    dedupe_filenames: bool,
    label_filter: str,
) -> List[Sample]:
    positive_set = {normalize_label(label) for label in positive_labels}
    negative_set = {normalize_label(label) for label in negative_labels}
    samples: List[Sample] = []
    seen_filenames: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_raw = row.get(label_column, "")
            filename_raw = row.get(filename_column, "")
            project_raw = row.get(project_column, "")
            if not label_raw or not filename_raw or not project_raw:
                continue
            label_norm = normalize_label(label_raw)
            if label_norm not in positive_set and label_norm not in negative_set:
                continue
            if label_filter == "positive" and label_norm not in positive_set:
                continue
            if label_filter == "negative" and label_norm not in negative_set:
                continue

            filename_norm = decode_percent_newlines(filename_raw, decode_newlines)
            if dedupe_filenames and filename_norm in seen_filenames:
                continue
            seen_filenames.add(filename_norm)

            image_path = image_root / project_raw / filename_norm
            if not image_path.exists():
                continue

            label = 1 if label_norm in positive_set else 0
            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    label_raw=label_raw,
                    filename=filename_norm,
                    project=project_raw,
                )
            )

    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate blur threshold review grids.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth.csv"),
        help="Path to the ground-truth CSV (default: ground_truth.csv).",
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--positive-labels", nargs="+", required=True)
    parser.add_argument("--negative-labels", nargs="+", required=True)
    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--project-column", type=str, default="project")
    parser.add_argument("--decode-percent-newlines", action="store_true")
    parser.add_argument("--dedupe-filenames", action="store_true", default=True)
    parser.add_argument("--no-dedupe-filenames", dest="dedupe_filenames", action="store_false")
    parser.add_argument(
        "--label-filter",
        choices=("all", "positive", "negative"),
        default="all",
        help="Restrict samples to only positive or negative labels.",
    )

    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--blur-values", nargs="+", type=float, default=[])
    parser.add_argument("--blur-range", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"))

    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    blur_values = [float(value) for value in args.blur_values]
    if args.blur_range:
        blur_values.extend(expand_range(args.blur_range))
    blur_values = sorted(set(blur_values))
    if not blur_values:
        raise ValueError("Provide --blur-values or --blur-range.")

    positive_labels = parse_label_list(args.positive_labels)
    negative_labels = parse_label_list(args.negative_labels)

    samples = load_samples(
        csv_path=args.csv_path,
        image_root=args.image_root,
        label_column=args.label_column,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        filename_column=args.filename_column,
        project_column=args.project_column,
        decode_newlines=args.decode_percent_newlines,
        dedupe_filenames=args.dedupe_filenames,
        label_filter=args.label_filter,
    )

    if not samples:
        raise ValueError("No samples found after filtering.")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    samples = samples[: max(1, args.max_images)]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = [
        "# Blur Threshold Review",
        "",
        f"Blur values: {', '.join(str(v) for v in blur_values)}",
        "",
        "Review each grid and decide:",
        "- a blur range that keeps the original label",
        "- a blur range that flips to \"not usable\"",
        "",
    ]

    for idx, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            images = [image]
            titles = ["orig"]
            for radius in blur_values:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                images.append(blurred)
                titles.append(f"blur r={radius:g}")

        cols = len(images)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2.8, 2.8))
        if cols == 1:
            axes = [axes]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        grid_name = f"blur_grid_{idx:02d}.png"
        grid_path = output_dir / grid_name
        fig.tight_layout()
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)

        md_lines.append(f"## Sample {idx}")
        md_lines.append(f"- Project: `{sample.project}`")
        md_lines.append(f"- Filename: `{sample.filename}`")
        md_lines.append(f"- Label: `{sample.label_raw}`")
        md_lines.append("")
        md_lines.append(f"![]({grid_name})")
        md_lines.append("")

    md_path = output_dir / "blur_threshold_review.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved {len(samples)} grids to: {output_dir}")
    print(f"Review file: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
