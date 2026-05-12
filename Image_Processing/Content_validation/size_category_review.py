#!/usr/bin/env python3
"""
Generate a size-category review grid for blur thresholds.

This tool groups images by pixel count and saves a 1x3 grid with one example
for each category (small/medium/very_large). Use this to validate the
size buckets for blur label switching.

How to run (example):
  uv run python Image_Processing/Content_validation/size_category_review.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --project-column project \
    --filename-column filename \
    --label-column "usability considering nfp" \
    --positive-labels "usable" \
    --negative-labels "not usable" \
    --decode-percent-newlines \
    --output-dir Image_Processing/Content_validation/size_category_review
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image


@dataclass
class Sample:
    image_path: Path
    label_raw: str
    filename: str
    project: str
    width: int
    height: int
    pixel_count: int


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


def categorize_size(pixel_count: int, thresholds_mp: Tuple[float, float]) -> str:
    size_mp = pixel_count / 1_000_000
    small_max, medium_max = thresholds_mp
    if size_mp <= small_max:
        return "small"
    if size_mp <= medium_max:
        return "medium"
    return "very_large"


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

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except OSError:
                continue
            pixel_count = width * height

            samples.append(
                Sample(
                    image_path=image_path,
                    label_raw=label_raw,
                    filename=filename_norm,
                    project=project_raw,
                    width=width,
                    height=height,
                    pixel_count=pixel_count,
                )
            )

    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pixel-size category review grid.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blur-size-small-max-mp", type=float, default=2.0)
    parser.add_argument("--blur-size-medium-max-mp", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def format_pixels(width: int, height: int, pixel_count: int) -> str:
    size_mp = pixel_count / 1_000_000
    return f"{width}x{height} ({size_mp:.2f} MP)"


def main() -> int:
    args = build_parser().parse_args()

    thresholds = (
        args.blur_size_small_max_mp,
        args.blur_size_medium_max_mp,
    )
    if not (thresholds[0] < thresholds[1]):
        raise ValueError("Size thresholds must be strictly increasing.")

    samples = load_samples(
        csv_path=args.csv_path,
        image_root=args.image_root,
        label_column=args.label_column,
        positive_labels=parse_label_list(args.positive_labels),
        negative_labels=parse_label_list(args.negative_labels),
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

    categories = ["small", "medium", "very_large"]
    selected: Dict[str, Sample] = {}
    counts: Dict[str, int] = {category: 0 for category in categories}

    for sample in samples:
        category = categorize_size(sample.pixel_count, thresholds)
        counts[category] += 1
        if category not in selected:
            selected[category] = sample
        if len(selected) == len(categories):
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(categories), figsize=(len(categories) * 3.2, 3.2))
    if len(categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        sample = selected.get(category)
        if sample is None:
            ax.text(0.5, 0.5, f"missing {category}", ha="center", va="center")
            ax.axis("off")
            continue
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
        ax.imshow(image)
        title = f"{category}\n{format_pixels(sample.width, sample.height, sample.pixel_count)}\n{sample.label_raw}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    grid_path = args.output_dir / "size_category_grid.png"
    fig.tight_layout()
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    md_lines = [
        "# Size Category Review",
        "",
        f"Thresholds (MP): small<= {thresholds[0]}, medium<= {thresholds[1]}, very_large> {thresholds[1]}",
        f"Label filter: {args.label_filter}",
        "",
        "Counts by category:",
    ]
    for category in categories:
        md_lines.append(f"- {category}: {counts.get(category, 0)}")
    md_lines.extend(["", "![](size_category_grid.png)", ""])

    for category in categories:
        sample = selected.get(category)
        if sample is None:
            md_lines.append(f"## {category}")
            md_lines.append("- Sample: not found")
            md_lines.append("")
            continue
        md_lines.append(f"## {category}")
        md_lines.append(f"- Project: `{sample.project}`")
        md_lines.append(f"- Filename: `{sample.filename}`")
        md_lines.append(f"- Label: `{sample.label_raw}`")
        md_lines.append(f"- Size: {format_pixels(sample.width, sample.height, sample.pixel_count)}")
        md_lines.append(f"- Path: `{sample.image_path}`")
        md_lines.append("")

    md_path = args.output_dir / "size_category_review.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved size category grid to: {grid_path}")
    print(f"Review file: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
