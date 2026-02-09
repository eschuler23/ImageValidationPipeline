#!/usr/bin/env python3
"""Filter a ground-truth CSV to rows whose image files exist under Images/.

This mirrors the training loader logic:
- label mapping (positive/negative)
- %0A decode for matching
- dedupe on decoded filename
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def normalize_label(label: str) -> str:
    return label.strip().lower()


def parse_label_list(values: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for value in values:
        for piece in value.split(","):
            cleaned = normalize_label(piece)
            if cleaned:
                labels.append(cleaned)
    return labels


def decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter ground-truth CSV to existing images.")
    parser.add_argument("--input", type=Path, default=Path("ground_truth.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("ground_truth_images_only.csv")
    )
    parser.add_argument("--image-root", type=Path, default=Path("Images"))
    parser.add_argument("--label-column", type=str, default="usability considering nfp")
    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--project-column", type=str, default="project")
    parser.add_argument("--positive-labels", nargs="+", default=["usable"])
    parser.add_argument("--negative-labels", nargs="+", default=["not usable"])
    parser.add_argument("--decode-percent-newlines", action="store_true", default=True)
    parser.add_argument("--no-decode-percent-newlines", dest="decode_percent_newlines", action="store_false")
    parser.add_argument("--dedupe-filenames", action="store_true", default=True)
    parser.add_argument("--no-dedupe-filenames", dest="dedupe_filenames", action="store_false")
    parser.add_argument("--report-missing-projects", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    positive_set = set(parse_label_list(args.positive_labels))
    negative_set = set(parse_label_list(args.negative_labels))
    if not positive_set or not negative_set:
        raise ValueError("Both positive and negative label lists are required.")

    stats = Counter()
    missing_by_project: Dict[str, int] = Counter()
    seen_filenames: set[str] = set()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        fieldnames = reader.fieldnames
        kept_rows: List[Dict[str, str]] = []

        for row in reader:
            stats["rows"] += 1
            label_raw = row.get(args.label_column, "")
            filename_raw = row.get(args.filename_column, "")
            filename_norm = decode_percent_newlines(filename_raw, args.decode_percent_newlines)
            if filename_norm != filename_raw:
                stats["decoded_percent_newlines"] += 1
            project_raw = row.get(args.project_column, "")

            if not label_raw:
                stats["missing_label"] += 1
                continue
            if not filename_raw:
                stats["missing_filename"] += 1
                continue
            if not project_raw:
                stats["missing_project"] += 1
                continue

            normalized = normalize_label(label_raw)
            if normalized in positive_set:
                label = 1
            elif normalized in negative_set:
                label = 0
            else:
                stats["unknown_label"] += 1
                continue

            if args.dedupe_filenames:
                if filename_norm in seen_filenames:
                    stats["duplicate_filename"] += 1
                    continue
                seen_filenames.add(filename_norm)

            image_path = args.image_root / project_raw / filename_norm
            if not image_path.exists():
                stats["missing_file"] += 1
                missing_by_project[project_raw] += 1
                continue

            stats["matched"] += 1
            kept_rows.append(row)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input rows: {stats['rows']}")
    print(f"Kept rows: {stats['matched']}")
    print(f"Missing files: {stats['missing_file']}")
    print(f"Duplicates dropped: {stats['duplicate_filename']}")
    print(f"Unknown labels: {stats['unknown_label']}")
    print(f"Decoded %0A rows: {stats['decoded_percent_newlines']}")
    if args.report_missing_projects and missing_by_project:
        print("\nTop missing-file projects:")
        for project, count in missing_by_project.most_common(args.report_missing_projects):
            print(f"- {project}: {count}")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
