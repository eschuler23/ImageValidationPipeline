"""
Loading step for the content-validation pipeline.

This module is intentionally small and highly readable. It does only two things:
1) Validate the image root on disk (so downstream steps can safely resolve paths).
2) Load ground-truth rows from the CSV without altering filenames or labels.

We keep this logic separate so the main pipeline can clearly show:
- first: where images live
- second: how labels are read

The functions return light summaries so the main script can log what was loaded
without mixing in training logic.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GroundTruthRow:
    """Single CSV row we care about in the pipeline."""

    row_index: int
    project: str
    filename: str
    label_raw: str
    raw_row: Dict[str, str]


def _decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def load_image_root(image_root: Path) -> Path:
    """
    Validate the image root path.

    This does not enumerate or open images; it only ensures the folder exists so
    downstream steps can resolve full paths safely.
    """

    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")
    return image_root


def load_ground_truth_rows(
    csv_path: Path,
    *,
    project_column: str,
    filename_column: str,
    label_column: str,
    decode_percent_newlines: bool = False,
) -> Tuple[List[GroundTruthRow], Dict[str, int]]:
    """
    Load ground-truth rows from the CSV.

    We do not alter filenames or labels except for an optional %0A -> newline
    decode. This keeps the loader honest and makes it easier to debug mismatches.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: List[GroundTruthRow] = []
    stats = {
        "rows": 0,
        "missing_project": 0,
        "missing_filename": 0,
        "missing_label": 0,
        "decoded_percent_newlines": 0,
    }

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            stats["rows"] += 1
            project = row.get(project_column, "")
            filename = row.get(filename_column, "")
            label_raw = row.get(label_column, "")

            if not project:
                stats["missing_project"] += 1
                continue
            if not filename:
                stats["missing_filename"] += 1
                continue
            if not label_raw:
                stats["missing_label"] += 1
                continue

            filename_decoded = _decode_percent_newlines(filename, decode_percent_newlines)
            if filename_decoded != filename:
                stats["decoded_percent_newlines"] += 1

            rows.append(
                GroundTruthRow(
                    row_index=idx,
                    project=project,
                    filename=filename_decoded,
                    label_raw=label_raw,
                    raw_row=dict(row),
                )
            )

    return rows, stats
