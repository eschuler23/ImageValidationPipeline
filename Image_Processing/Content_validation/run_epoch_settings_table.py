#!/usr/bin/env python3
"""
Generate a per-epoch validation score table with training settings.

How to run:
  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/run_epoch_settings_table.py \
    --runs-root Image_Processing/Content_validation/runs \
    --output-csv reports/run_epoch_settings_table.csv \
    --output-md reports/run_epoch_settings_table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SEED_RE = re.compile(r"(?:^|[_-])seed[-_]?([0-9]+)", re.IGNORECASE)
LR_RE = re.compile(r"(?:^|[_-])lr-([^_]+)", re.IGNORECASE)
EPOCH_RE = re.compile(r"(?:^|[_-])ep(?:ochs?)?[-_]?([0-9]+)", re.IGNORECASE)
FREEZE_RE = re.compile(r"(?:^|[_-])freeze[-_]?([0-9]+)", re.IGNORECASE)

CSV_COLUMNS = [
    "run",
    "model",
    "weights",
    "epoch",
    "val_f1",
    "val_acc",
    "train_loss",
    "seed",
    "lr",
    "unfreeze_lr",
    "weight_decay",
    "batch_size",
    "grad_accum_steps",
    "freeze_backbone_epochs",
    "epochs_planned",
    "amp",
    "random_augmentation",
    "csv_path",
    "val_base",
    "test_base",
    "val_samples",
    "test_samples",
    "inferred_fields",
]

MD_COLUMNS = [
    "run",
    "model",
    "weights",
    "epoch",
    "val_f1",
    "val_acc",
    "seed",
    "lr",
    "weight_decay",
    "batch_size",
    "freeze_backbone_epochs",
    "epochs_planned",
    "inferred_fields",
]


@dataclass(frozen=True)
class RunConfig:
    run: str
    csv_path: Optional[str]
    random_augmentation: Optional[bool]
    training: Dict[str, Any]
    splits: Dict[str, Any]


@dataclass(frozen=True)
class MetricsInfo:
    model: str
    weights: Optional[str]
    val_samples: Optional[int]
    test_samples: Optional[int]
    history: List[Dict[str, Any]]


def _parse_seed(run_name: str, training: Dict[str, Any]) -> Optional[int]:
    seed_value = training.get("seed")
    if seed_value is not None:
        try:
            return int(seed_value)
        except (TypeError, ValueError):
            pass

    match = SEED_RE.search(run_name)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _parse_lr(run_name: str) -> Optional[float]:
    match = LR_RE.search(run_name)
    if not match:
        return None
    token = match.group(1).replace("p", ".").replace("P", ".")
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _parse_epochs(run_name: str) -> Optional[int]:
    match = EPOCH_RE.search(run_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_freeze_epochs(run_name: str) -> Optional[int]:
    match = FREEZE_RE.search(run_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: invalid JSON in {path}")
        return None


def _load_run_config(run_dir: Path) -> Optional[RunConfig]:
    preprocessing_path = run_dir / "preprocessing.json"
    data = _load_json(preprocessing_path)
    if data is None:
        return None
    return RunConfig(
        run=run_dir.name,
        csv_path=data.get("csv_path"),
        random_augmentation=data.get("random_augmentation"),
        training=data.get("training", {}) if isinstance(data.get("training"), dict) else {},
        splits=data.get("splits", {}) if isinstance(data.get("splits"), dict) else {},
    )


def _load_metrics(metrics_path: Path) -> Optional[MetricsInfo]:
    data = _load_json(metrics_path)
    if data is None:
        return None
    history = data.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    model = data.get("model") or metrics_path.parent.name
    weights = data.get("weights")
    val_samples = data.get("val_samples")
    test_samples = data.get("test_samples")
    return MetricsInfo(
        model=model,
        weights=weights,
        val_samples=val_samples,
        test_samples=test_samples,
        history=history,
    )


def _iter_metrics(run_dir: Path) -> Iterable[tuple[Path, MetricsInfo]]:
    for metrics_path in sorted(run_dir.glob("*/metrics.json")):
        metrics = _load_metrics(metrics_path)
        if metrics is None:
            continue
        yield metrics_path, metrics


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_rows(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        run_config = _load_run_config(run_dir)
        if run_config is None:
            continue

        training = run_config.training
        splits = run_config.splits
        inferred_fields: List[str] = []
        seed = _parse_seed(run_config.run, training)
        if seed is not None and training.get("seed") is None:
            inferred_fields.append("seed")

        lr = training.get("lr")
        if lr is None:
            lr = _parse_lr(run_config.run)
            if lr is not None:
                inferred_fields.append("lr")

        epochs_planned = training.get("epochs")
        if epochs_planned is None:
            epochs_planned = _parse_epochs(run_config.run)
            if epochs_planned is not None:
                inferred_fields.append("epochs")

        freeze_backbone_epochs = training.get("freeze_backbone_epochs")
        if freeze_backbone_epochs is None:
            freeze_backbone_epochs = _parse_freeze_epochs(run_config.run)
            if freeze_backbone_epochs is not None:
                inferred_fields.append("freeze_backbone_epochs")

        inferred_fields_text = ",".join(inferred_fields)

        for _, metrics in _iter_metrics(run_dir):
            for entry in metrics.history:
                if not isinstance(entry, dict):
                    continue
                epoch_raw = entry.get("epoch")
                try:
                    epoch = int(epoch_raw)
                except (TypeError, ValueError):
                    continue

                row = {
                    "run": run_config.run,
                    "model": metrics.model,
                    "weights": metrics.weights,
                    "epoch": epoch,
                    "val_f1": _coerce_number(entry.get("f1")),
                    "val_acc": _coerce_number(entry.get("accuracy")),
                    "train_loss": _coerce_number(entry.get("train_loss")),
                    "seed": seed,
                    "lr": lr,
                    "unfreeze_lr": training.get("unfreeze_lr"),
                    "weight_decay": training.get("weight_decay"),
                    "batch_size": training.get("batch_size"),
                    "grad_accum_steps": training.get("grad_accum_steps"),
                    "freeze_backbone_epochs": freeze_backbone_epochs,
                    "epochs_planned": epochs_planned,
                    "amp": training.get("amp"),
                    "random_augmentation": run_config.random_augmentation,
                    "csv_path": run_config.csv_path,
                    "val_base": splits.get("val_base"),
                    "test_base": splits.get("test_base"),
                    "val_samples": metrics.val_samples,
                    "test_samples": metrics.test_samples,
                    "inferred_fields": inferred_fields_text,
                }
                rows.append(row)

    rows.sort(key=lambda row: (row["run"], row["model"], row["epoch"]))
    return rows


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([row.get(col) for col in CSV_COLUMNS])


def _write_md(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(MD_COLUMNS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(MD_COLUMNS)) + " |\n")
        for row in rows:
            formatted = [_format_md_value(row.get(col)) for col in MD_COLUMNS]
            handle.write("| " + " | ".join(formatted) + " |\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-epoch validation score table with training settings."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/run_epoch_settings_table.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/run_epoch_settings_table.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _build_rows(args.runs_root)
    if not rows:
        raise ValueError("No rows found. Check runs root and metrics.json files.")

    _write_csv(rows, args.output_csv)
    _write_md(rows, args.output_md)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote {len(rows)} rows to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
