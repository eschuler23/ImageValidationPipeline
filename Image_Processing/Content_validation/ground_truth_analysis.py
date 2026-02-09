#!/usr/bin/env python3
"""
Ground-truth label analysis utility.

This helper provides a quick overview of label distributions (counts +
percentages) so we can understand the dataset before training. It also supports
subset analysis by filename, which will be useful later when we slice
confusion-matrix outcomes (false positives/negatives) and inspect other label
columns like "Blurr" or "usability considering blur".

How to run (NFP usability example):
  uv run python Image_Processing/Content_validation/ground_truth_analysis.py \
    --csv-path ground_truth.csv \
    --label-columns "usability considering nfp" "usability considering blur" "Blurr" \
    --usability-column "usability considering nfp" \
    --usable-labels "usable" \
    --unusable-labels "not usable" \
    --decode-percent-newlines

How to run (blur usability example):
  uv run python Image_Processing/Content_validation/ground_truth_analysis.py \
    --csv-path ground_truth.csv \
    --label-columns "usability considering blur" "Blurr" \
    --usability-column "usability considering blur" \
    --usable-labels "focused enough" \
    --unusable-labels "too blurry" \
    --decode-percent-newlines
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GroundTruthRow:
    """Single CSV row plus normalized filename for matching."""

    row_number: int
    filename_raw: str
    filename: str
    project: str
    row: Dict[str, str]


@dataclass(frozen=True)
class LabelBreakdown:
    label: str
    count: int
    percent_of_total: float
    percent_of_labeled: float


@dataclass(frozen=True)
class ColumnSummary:
    column: str
    total_rows: int
    labeled_rows: int
    missing_rows: int
    missing_percent: float
    labels: Tuple[LabelBreakdown, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "column": self.column,
            "total_rows": self.total_rows,
            "labeled_rows": self.labeled_rows,
            "missing_rows": self.missing_rows,
            "missing_percent": self.missing_percent,
            "labels": [
                {
                    "label": label.label,
                    "count": label.count,
                    "percent_of_total": label.percent_of_total,
                    "percent_of_labeled": label.percent_of_labeled,
                }
                for label in self.labels
            ],
        }


def decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace("_", " ")
    return " ".join(cleaned.split())


def is_missing(label: str, *, missing_values: Sequence[str], normalize: bool) -> bool:
    if label is None:
        return True
    cleaned = label.strip()
    if not cleaned:
        return True
    if normalize:
        cleaned = normalize_label(cleaned)
    return cleaned in missing_values


class GroundTruthAnalyzer:
    """Utility for computing label distributions from a ground-truth CSV."""

    def __init__(
        self,
        rows: Sequence[GroundTruthRow],
        *,
        filename_column: str,
        project_column: str,
        source_path: Optional[Path] = None,
    ) -> None:
        self._rows = list(rows)
        self.filename_column = filename_column
        self.project_column = project_column
        self.source_path = source_path

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        *,
        filename_column: str = "filename",
        project_column: str = "project",
        decode_percent_newlines_enabled: bool = False,
        dedupe_filenames: bool = False,
    ) -> "GroundTruthAnalyzer":
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        rows: List[GroundTruthRow] = []
        seen_filenames: set[str] = set()

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                filename_raw = row.get(filename_column, "")
                filename = decode_percent_newlines(
                    filename_raw,
                    decode_percent_newlines_enabled,
                )

                if dedupe_filenames and filename:
                    if filename in seen_filenames:
                        continue
                    seen_filenames.add(filename)

                rows.append(
                    GroundTruthRow(
                        row_number=row_number,
                        filename_raw=filename_raw,
                        filename=filename,
                        project=row.get(project_column, ""),
                        row=dict(row),
                    )
                )

        return cls(
            rows,
            filename_column=filename_column,
            project_column=project_column,
            source_path=csv_path,
        )

    @property
    def rows(self) -> Tuple[GroundTruthRow, ...]:
        return tuple(self._rows)

    @property
    def total_rows(self) -> int:
        return len(self._rows)

    def columns(self) -> Tuple[str, ...]:
        columns: set[str] = set()
        for row in self._rows:
            columns.update(row.row.keys())
        return tuple(sorted(columns))

    def subset_by_filenames(self, filenames: Iterable[str]) -> "GroundTruthAnalyzer":
        filename_set = {name.strip() for name in filenames if name and name.strip()}
        subset = [row for row in self._rows if row.filename in filename_set]
        return GroundTruthAnalyzer(
            subset,
            filename_column=self.filename_column,
            project_column=self.project_column,
            source_path=self.source_path,
        )

    def subset_by_predicate(
        self,
        predicate: Callable[[GroundTruthRow], bool],
    ) -> "GroundTruthAnalyzer":
        subset = [row for row in self._rows if predicate(row)]
        return GroundTruthAnalyzer(
            subset,
            filename_column=self.filename_column,
            project_column=self.project_column,
            source_path=self.source_path,
        )

    def summarize_labels(
        self,
        label_column: str,
        *,
        normalize: bool = True,
        missing_values: Optional[Sequence[str]] = None,
        sort_by_count: bool = True,
    ) -> ColumnSummary:
        missing_values = tuple(
            normalize_label(value) if normalize else value
            for value in (missing_values or ())
        )
        label_counts: Counter[str] = Counter()
        missing_rows = 0

        for row in self._rows:
            raw_value = row.row.get(label_column, "")
            if is_missing(raw_value, missing_values=missing_values, normalize=normalize):
                missing_rows += 1
                continue

            label = normalize_label(raw_value) if normalize else raw_value.strip()
            label_counts[label] += 1

        total_rows = len(self._rows)
        labeled_rows = total_rows - missing_rows
        missing_percent = 100.0 * missing_rows / total_rows if total_rows else 0.0

        if sort_by_count:
            items = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        else:
            items = label_counts.items()

        labels = tuple(
            LabelBreakdown(
                label=label,
                count=count,
                percent_of_total=100.0 * count / total_rows if total_rows else 0.0,
                percent_of_labeled=100.0 * count / labeled_rows if labeled_rows else 0.0,
            )
            for label, count in items
        )

        return ColumnSummary(
            column=label_column,
            total_rows=total_rows,
            labeled_rows=labeled_rows,
            missing_rows=missing_rows,
            missing_percent=missing_percent,
            labels=labels,
        )

    def summarize_usability(
        self,
        label_column: str,
        *,
        usable_labels: Sequence[str],
        unusable_labels: Sequence[str],
        normalize: bool = True,
        missing_values: Optional[Sequence[str]] = None,
        include_unknown: bool = True,
        include_zero: bool = True,
    ) -> ColumnSummary:
        usable_set = {normalize_label(label) for label in usable_labels} if normalize else set(usable_labels)
        unusable_set = (
            {normalize_label(label) for label in unusable_labels} if normalize else set(unusable_labels)
        )
        missing_values = tuple(
            normalize_label(value) if normalize else value
            for value in (missing_values or ())
        )

        counts: Counter[str] = Counter()
        missing_rows = 0

        for row in self._rows:
            raw_value = row.row.get(label_column, "")
            if is_missing(raw_value, missing_values=missing_values, normalize=normalize):
                missing_rows += 1
                continue

            label = normalize_label(raw_value) if normalize else raw_value.strip()
            if label in usable_set:
                counts["usable"] += 1
            elif label in unusable_set:
                counts["unusable"] += 1
            elif include_unknown:
                counts["unknown"] += 1

        total_rows = len(self._rows)
        labeled_rows = total_rows - missing_rows
        missing_percent = 100.0 * missing_rows / total_rows if total_rows else 0.0

        ordered_labels = ["usable", "unusable"]
        if include_unknown:
            ordered_labels.append("unknown")

        labels: List[LabelBreakdown] = []
        for name in ordered_labels:
            count = counts.get(name, 0)
            if count == 0 and not include_zero:
                continue
            labels.append(
                LabelBreakdown(
                    label=name,
                    count=count,
                    percent_of_total=100.0 * count / total_rows if total_rows else 0.0,
                    percent_of_labeled=100.0 * count / labeled_rows if labeled_rows else 0.0,
                )
            )

        return ColumnSummary(
            column=label_column,
            total_rows=total_rows,
            labeled_rows=labeled_rows,
            missing_rows=missing_rows,
            missing_percent=missing_percent,
            labels=tuple(labels),
        )

    def summarize_columns(
        self,
        label_columns: Sequence[str],
        *,
        normalize: bool = True,
        missing_values: Optional[Sequence[str]] = None,
    ) -> Dict[str, ColumnSummary]:
        return {
            column: self.summarize_labels(
                column,
                normalize=normalize,
                missing_values=missing_values,
            )
            for column in label_columns
        }


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def print_column_summary(summary: ColumnSummary) -> None:
    print(f"\nColumn: {summary.column}")
    print(f"- Total rows: {summary.total_rows}")
    print(
        f"- Missing labels: {summary.missing_rows} ({format_percent(summary.missing_percent)})"
    )
    for label in summary.labels:
        print(
            "- "
            f"{label.label}: {label.count} "
            f"({format_percent(label.percent_of_total)} of total, "
            f"{format_percent(label.percent_of_labeled)} of labeled)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize ground-truth label distributions.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth.csv"),
        help="Path to the ground-truth CSV (default: ground_truth.csv).",
    )
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=["usability considering nfp", "usability considering blur", "Blurr"],
        help="Label columns to summarize (space-separated).",
    )
    parser.add_argument(
        "--usability-column",
        type=str,
        default="usability considering nfp",
        help="Column to use for usable/unusable summary.",
    )
    parser.add_argument("--usable-labels", nargs="+", default=["usable"])
    parser.add_argument("--unusable-labels", nargs="+", default=["not usable"])
    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--project-column", type=str, default="project")
    parser.add_argument("--decode-percent-newlines", action="store_true")
    parser.add_argument("--dedupe-filenames", action="store_true", default=False)
    parser.add_argument(
        "--no-normalize-labels",
        dest="normalize_labels",
        action="store_false",
        help="Keep label strings exactly as they appear in the CSV.",
    )
    parser.set_defaults(normalize_labels=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write summary JSON for reuse in reports.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    analyzer = GroundTruthAnalyzer.from_csv(
        args.csv_path,
        filename_column=args.filename_column,
        project_column=args.project_column,
        decode_percent_newlines_enabled=args.decode_percent_newlines,
        dedupe_filenames=args.dedupe_filenames,
    )

    print(f"Loaded {analyzer.total_rows} ground-truth rows.")

    available_columns = set(analyzer.columns())
    summaries: Dict[str, ColumnSummary] = {}

    for column in args.label_columns:
        if column not in available_columns:
            print(f"\nColumn: {column}")
            print("- Warning: column not found in CSV; skipping.")
            continue
        summary = analyzer.summarize_labels(column, normalize=args.normalize_labels)
        summaries[column] = summary
        print_column_summary(summary)

    if args.usability_column:
        if args.usability_column not in available_columns:
            print(f"\nUsability summary: {args.usability_column}")
            print("- Warning: column not found in CSV; skipping.")
        else:
            usability_summary = analyzer.summarize_usability(
                args.usability_column,
                usable_labels=args.usable_labels,
                unusable_labels=args.unusable_labels,
                normalize=args.normalize_labels,
            )
            print("\nUsability summary:")
            print_column_summary(usability_summary)
            summaries[f"usability::{args.usability_column}"] = usability_summary

    if args.output_json:
        payload = {
            "csv_path": str(args.csv_path),
            "total_rows": analyzer.total_rows,
            "summaries": {key: summary.to_dict() for key, summary in summaries.items()},
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved summary JSON to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
