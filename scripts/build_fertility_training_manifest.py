#!/usr/bin/env python3
"""Build a training manifest by matching CSV filenames inside an image tree."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_labels(values: Iterable[str]) -> set[str]:
    labels: set[str] = set()
    for value in values:
        for piece in value.split(","):
            cleaned = piece.strip().lower()
            if cleaned:
                labels.add(cleaned)
    return labels


def index_images(image_root: Path) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_file():
            by_name[path.name].append(path)
    return by_name


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[str]]:
    wanted_labels = parse_labels(args.labels)
    image_index = index_images(args.image_root)
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {args.filename_column, args.label_column}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV is missing columns: {sorted(missing_columns)}")

        for source_row in reader:
            raw_label = source_row.get(args.label_column, "")
            label = raw_label.strip().lower()
            filename_raw = source_row.get(args.filename_column, "")
            filename = filename_raw.strip()
            if label not in wanted_labels or not filename:
                continue

            hits = image_index.get(filename, [])
            if not hits:
                missing.append(
                    {
                        "filename": filename,
                        "label": raw_label,
                    }
                )
                continue

            if len(hits) > 1 and not args.allow_duplicate_filenames:
                hit_list = ", ".join(str(path) for path in hits)
                raise ValueError(f"Duplicate filename match for {filename}: {hit_list}")

            image_path = hits[0]
            relative_parent = image_path.parent.relative_to(args.image_root)
            row = {
                "project": str(relative_parent),
                "filename": image_path.name,
                args.label_column: raw_label,
                "source_csv_filename": filename_raw,
                "image_path": str(image_path),
            }
            rows.append(row)
            counts[label] += 1

    return rows, missing, counts


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--missing-csv", type=Path)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--allow-duplicate-filenames",
        action="store_true",
        help="Use the first match when the same filename appears more than once.",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    rows, missing, counts = build_manifest(args)
    fieldnames = [
        "project",
        "filename",
        args.label_column,
        "source_csv_filename",
        "image_path",
    ]
    write_csv(args.output_csv, rows, fieldnames)

    if args.missing_csv:
        write_csv(args.missing_csv, missing, ["filename", "label"])

    print(f"Matched rows: {len(rows)}")
    for label, count in sorted(counts.items()):
        print(f"- {label}: {count}")
    print(f"Missing rows: {len(missing)}")
    print(f"Output CSV: {args.output_csv}")
    if args.missing_csv:
        print(f"Missing CSV: {args.missing_csv}")


if __name__ == "__main__":
    main()
