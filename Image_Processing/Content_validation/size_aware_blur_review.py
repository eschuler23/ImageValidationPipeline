#!/usr/bin/env python3
"""
Generate blur-augmented grids for size-aware blur thresholds.

This tool picks one image per pixel-size category (small/medium/very_large),
then applies a blur range and labels each blur as keep/flip based on the
size-aware thresholds.

How to run (example):
  uv run python Image_Processing/Content_validation/size_aware_blur_review.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --project-column project \
    --filename-column filename \
    --label-column "usability considering nfp" \
    --positive-labels "usable" \
    --negative-labels "not usable" \
    --decode-percent-newlines \
    --label-filter positive \
    --blur-range 0 20 2 \
    --output-dir Image_Processing/Content_validation/size_aware_blur_review
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


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


def categorize_size(pixel_count: int, thresholds_mp: Tuple[float, float]) -> str:
    size_mp = pixel_count / 1_000_000
    small_max, medium_max = thresholds_mp
    if size_mp <= small_max:
        return "small"
    if size_mp <= medium_max:
        return "medium"
    return "very_large"


def pick_samples(
    samples: List[Sample],
    thresholds_mp: Tuple[float, float],
    seed: int,
) -> Dict[str, Sample]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    selected: Dict[str, Sample] = {}
    for sample in samples:
        category = categorize_size(sample.pixel_count, thresholds_mp)
        if category not in selected:
            selected[category] = sample
        if len(selected) == 3:
            break
    return selected


def pick_sample_for_category(
    samples: List[Sample],
    thresholds_mp: Tuple[float, float],
    category: str,
    sample_index: int,
) -> Sample:
    filtered: List[Sample] = [
        sample for sample in samples if categorize_size(sample.pixel_count, thresholds_mp) == category
    ]
    if not filtered:
        raise ValueError(f"No samples found for category '{category}'.")
    filtered.sort(key=lambda sample: str(sample.image_path))
    if sample_index < 0 or sample_index >= len(filtered):
        raise ValueError(
            f"sample-index {sample_index} out of range for '{category}' (0-{len(filtered) - 1})."
        )
    return filtered[sample_index]


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


def format_pixels(width: int, height: int, pixel_count: int) -> str:
    size_mp = pixel_count / 1_000_000
    return f"{width}x{height} ({size_mp:.2f} MP)"


def resolve_threshold(category: str, switches: Tuple[float, float, float]) -> float:
    small, medium, large = switches
    if category == "small":
        return small
    if category == "medium":
        return medium
    return large


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate size-aware blur review grids.")
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
    parser.add_argument("--blur-switch-small", type=float, default=2.0)
    parser.add_argument("--blur-switch-medium", type=float, default=10.0)
    parser.add_argument("--blur-switch-large", type=float, default=20.0)
    parser.add_argument("--blur-values", nargs="+", type=float, default=[])
    parser.add_argument("--blur-range", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument(
        "--category",
        choices=("all", "small", "medium", "very_large"),
        default="all",
        help="Restrict the review to a single size category.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="When using --category, choose which sample index to display.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    thresholds = (
        args.blur_size_small_max_mp,
        args.blur_size_medium_max_mp,
    )
    if not (thresholds[0] < thresholds[1]):
        raise ValueError("Size thresholds must be strictly increasing.")

    blur_values = [float(value) for value in args.blur_values]
    if args.blur_range:
        blur_values.extend(expand_range(args.blur_range))
    blur_values = sorted(set(blur_values))
    if not blur_values:
        raise ValueError("Provide --blur-values or --blur-range.")

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

    if args.category == "all":
        selected = pick_samples(samples, thresholds, args.seed)
        categories = ["small", "medium", "very_large"]
    else:
        sample = pick_sample_for_category(samples, thresholds, args.category, args.sample_index)
        selected = {args.category: sample}
        categories = [args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Size-Aware Blur Review",
        "",
        f"Thresholds (MP): small<= {thresholds[0]}, medium<= {thresholds[1]}, very_large> {thresholds[1]}",
        (
            "Blur switch radii: "
            f"small={args.blur_switch_small}, medium={args.blur_switch_medium}, "
            f"very_large={args.blur_switch_large}"
        ),
        "",
    ]

    switches = (
        args.blur_switch_small,
        args.blur_switch_medium,
        args.blur_switch_large,
    )

    for category in categories:
        sample = selected.get(category)
        if sample is None:
            md_lines.append(f"## {category}")
            md_lines.append("- Sample: not found")
            md_lines.append("")
            continue

        threshold = resolve_threshold(category, switches)
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            images = [image]
            titles = ["orig"]
            for radius in blur_values:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                label_tag = "keep" if radius <= threshold else "flip"
                images.append(blurred)
                titles.append(f"r={radius:g} {label_tag}")

        cols = len(images)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2.6, 2.6))
        if cols == 1:
            axes = [axes]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        grid_name = f"blur_{category}.png"
        grid_path = args.output_dir / grid_name
        fig.tight_layout()
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)

        md_lines.append(f"## {category}")
        md_lines.append(f"- Project: `{sample.project}`")
        md_lines.append(f"- Filename: `{sample.filename}`")
        md_lines.append(f"- Label: `{sample.label_raw}`")
        md_lines.append(f"- Size: {format_pixels(sample.width, sample.height, sample.pixel_count)}")
        md_lines.append(f"- Blur flip threshold: {threshold}")
        md_lines.append("")
        md_lines.append(f"![]({grid_name})")
        md_lines.append("")

    md_path = args.output_dir / "size_aware_blur_review.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved blur grids to: {args.output_dir}")
    print(f"Review file: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
