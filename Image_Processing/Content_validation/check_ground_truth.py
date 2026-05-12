#!/usr/bin/env python3
"""
Ground-truth completeness check for content validation.

This compares CSV entries against the on-disk Images/ folder and reports only
mismatches (missing files or missing labels). Filenames are kept exactly as-is
(spaces, casing, and special characters preserved).

How to run (from repo root):
  uv run python Image_Processing/Content_validation/check_ground_truth.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --label-column "usability considering nfp" \
    --project-column project \
    --filename-column filename

Optional mapping for project names (CSV -> folder):
  uv run python Image_Processing/Content_validation/check_ground_truth.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --label-column "usability considering nfp" \
    --project-column project \
    --filename-column filename \
    --project-map Image_Processing/Content_validation/project_map.json

Filename-only matching (ignore folder names):
  uv run python Image_Processing/Content_validation/check_ground_truth.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --label-column "usability considering nfp" \
    --filename-column filename \
    --match-mode filename

Filename-only matching with %0A -> newline decoding:
  uv run python Image_Processing/Content_validation/check_ground_truth.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --label-column "usability considering nfp" \
    --filename-column filename \
    --match-mode filename \
    --decode-percent-newlines
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class CsvEntry:
    project_raw: str
    project_mapped: str
    filename_raw: str
    filename_norm: str
    label_raw: str
    row_number: int


def parse_extensions(values: Sequence[str]) -> Set[str]:
    extensions: Set[str] = set()
    for value in values:
        for piece in value.split(","):
            cleaned = piece.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            extensions.add(cleaned)
    return extensions


def load_project_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Project map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Project map JSON must be an object mapping CSV project -> folder name.")
    return {str(k): str(v) for k, v in payload.items()}


def decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def iter_image_entries(
    image_root: Path,
    extensions: Set[str],
    ignore_hidden: bool,
) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for project_dir in sorted(image_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for file_path in sorted(project_dir.iterdir()):
            if not file_path.is_file():
                continue
            if ignore_hidden and file_path.name.startswith("."):
                continue
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            entries.append((project_dir.name, file_path.name))
    return entries


def format_entry(entry: CsvEntry) -> str:
    filename = entry.filename_norm or entry.filename_raw
    suffix = ""
    if entry.filename_raw and entry.filename_norm != entry.filename_raw:
        suffix = f" (raw: {entry.filename_raw})"
    return f"{entry.project_raw} / {filename}{suffix} (row {entry.row_number})"


def format_pair(pair: Tuple[str, str]) -> str:
    project, filename = pair
    return f"{project} / {filename}"


def print_section(title: str, entries: Sequence[str], limit: int) -> None:
    if not entries:
        return
    print(f"\n{title} ({len(entries)}):")
    for row in entries[:limit]:
        print(f"- {row}")
    if len(entries) > limit:
        print(f"  ... and {len(entries) - limit} more")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check CSV labels against Images/ on disk.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth.csv"),
        help="Path to the ground-truth CSV (default: ground_truth.csv).",
    )
    parser.add_argument("--image-root", type=Path, required=True, help="Root folder containing project subfolders.")
    parser.add_argument("--label-column", type=str, required=True, help="CSV column with labels (e.g., usability).")
    parser.add_argument("--project-column", type=str, default="project", help="CSV column with project/folder name.")
    parser.add_argument("--filename-column", type=str, default="filename", help="CSV column with filenames.")
    parser.add_argument("--project-map", type=Path, help="Optional JSON mapping of CSV project -> folder name.")
    parser.add_argument(
        "--match-mode",
        choices=["project", "filename"],
        default="filename",
        help="Match by project+filename or by filename only (default: filename).",
    )
    parser.add_argument(
        "--decode-percent-newlines",
        action="store_true",
        help="Decode %0A in CSV filenames into literal newlines before matching.",
    )
    parser.add_argument(
        "--dedupe-filenames",
        action="store_true",
        default=True,
        help="Drop duplicate filenames (keep first occurrence) during filename-only matching.",
    )
    parser.add_argument(
        "--no-dedupe-filenames",
        dest="dedupe_filenames",
        action="store_false",
        help="Keep duplicate filenames during filename-only matching.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions to treat as images (default: jpg/jpeg/png/bmp/tif/tiff/webp).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include dotfiles like .DS_Store.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=25,
        help="Max examples to print per mismatch category.",
    )
    parser.add_argument(
        "--quiet-ok",
        action="store_true",
        help="Suppress success output when no mismatches are found.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    extensions = parse_extensions(args.extensions)
    project_map = load_project_map(args.project_map)

    csv_entries: List[CsvEntry] = []
    missing_label_rows: List[CsvEntry] = []
    missing_project_rows: List[CsvEntry] = []
    missing_filename_rows: List[CsvEntry] = []
    decoded_count = 0

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            label_raw = row.get(args.label_column, "")
            project_raw = row.get(args.project_column, "")
            filename_raw = row.get(args.filename_column, "")
            filename_norm = decode_percent_newlines(filename_raw, args.decode_percent_newlines)
            if filename_norm != filename_raw:
                decoded_count += 1

            entry = CsvEntry(
                project_raw=project_raw,
                project_mapped=project_map.get(project_raw, project_raw),
                filename_raw=filename_raw,
                filename_norm=filename_norm,
                label_raw=label_raw,
                row_number=row_number,
            )
            csv_entries.append(entry)

            if not label_raw:
                missing_label_rows.append(entry)
            if args.match_mode == "project" and not project_raw:
                missing_project_rows.append(entry)
            if not filename_raw:
                missing_filename_rows.append(entry)

    image_entries = iter_image_entries(args.image_root, extensions, not args.include_hidden)

    csv_duplicates: List[str] = []
    image_duplicates: List[str] = []
    missing_files: List[CsvEntry] = []
    missing_labels: List[Tuple[str, str]] = []

    if args.match_mode == "project":
        csv_pairs = [(entry.project_mapped, entry.filename_norm) for entry in csv_entries]
        csv_set = set(csv_pairs)
        image_set = set(image_entries)
        csv_duplicates = [format_pair(pair) for pair, count in Counter(csv_pairs).items() if count > 1]
        missing_files = [
            entry
            for entry in csv_entries
            if (entry.project_mapped, entry.filename_norm) not in image_set
        ]
        missing_labels = [pair for pair in image_entries if pair not in csv_set]
    else:
        csv_filenames = [entry.filename_norm for entry in csv_entries if entry.filename_norm]
        image_filenames = [filename for _, filename in image_entries]
        csv_duplicates = [f"{name} (count {count})" for name, count in Counter(csv_filenames).items() if count > 1]
        image_duplicates = [f"{name} (count {count})" for name, count in Counter(image_filenames).items() if count > 1]

        if args.dedupe_filenames:
            seen_csv: set[str] = set()
            csv_entries_for_match: List[CsvEntry] = []
            csv_filenames_for_match: List[str] = []
            for entry in csv_entries:
                if not entry.filename_norm:
                    continue
                if entry.filename_norm in seen_csv:
                    continue
                seen_csv.add(entry.filename_norm)
                csv_entries_for_match.append(entry)
                csv_filenames_for_match.append(entry.filename_norm)

            seen_img: set[str] = set()
            image_entries_for_match: List[Tuple[str, str]] = []
            for pair in image_entries:
                filename = pair[1]
                if filename in seen_img:
                    continue
                seen_img.add(filename)
                image_entries_for_match.append(pair)
        else:
            csv_entries_for_match = [entry for entry in csv_entries if entry.filename_norm]
            csv_filenames_for_match = csv_filenames
            image_entries_for_match = image_entries

        csv_set = set(csv_filenames_for_match)
        image_set = {filename for _, filename in image_entries_for_match}
        missing_files = [
            entry
            for entry in csv_entries_for_match
            if entry.filename_norm not in image_set
        ]
        missing_labels = [
            pair
            for pair in image_entries_for_match
            if pair[1] not in csv_set
        ]

    has_issues = any(
        [
            missing_label_rows,
            missing_project_rows if args.match_mode == "project" else [],
            missing_filename_rows,
            csv_duplicates,
            image_duplicates,
            missing_files,
            missing_labels,
        ]
    )

    if not has_issues:
        if not args.quiet_ok:
            print(
                "OK: All CSV entries have matching files and all on-disk images have labels."
            )
        return 0

    print("Ground-truth completeness check found issues:")

    print_section(
        "CSV rows missing labels",
        [format_entry(entry) for entry in missing_label_rows],
        args.report_limit,
    )
    if args.match_mode == "project":
        print_section(
            "CSV rows missing project",
            [format_entry(entry) for entry in missing_project_rows],
            args.report_limit,
        )
    print_section(
        "CSV rows missing filename",
        [format_entry(entry) for entry in missing_filename_rows],
        args.report_limit,
    )
    if args.match_mode == "project":
        print_section(
            "Duplicate CSV entries (project/filename)",
            list(csv_duplicates),
            args.report_limit,
        )
    else:
        print_section(
            "Duplicate CSV filenames",
            list(csv_duplicates),
            args.report_limit,
        )
        print_section(
            "Duplicate image filenames across folders",
            list(image_duplicates),
            args.report_limit,
        )
    print_section(
        "CSV entries without matching files",
        [format_entry(entry) for entry in missing_files],
        args.report_limit,
    )
    print_section(
        "On-disk images without CSV rows",
        [format_pair(pair) for pair in missing_labels],
        args.report_limit,
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
