#!/usr/bin/env python3
"""
Run summary charts.

Parses the "Runs (recent)" table in `summary.md` and creates two scatter plots:
1) Epochs vs F1 (default: test_f1)
2) Epochs vs Accuracy (default: test_acc; optionally val_acc from metrics.json)

How to run:
  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/run_summary_charts.py \
    --summary-path summary.md \
    --output-dir reports/run_summary_charts

  uv run --python /Users/raven/Projects/Bachelors/.venv/bin/python \
    Image_Processing/Content_validation/run_summary_charts.py \
    --summary-path summary.md \
    --f1-metric val_f1 \
    --acc-metric val_acc \
    --output-dir reports/run_summary_charts
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass(frozen=True)
class RunRecord:
    run: str
    model: str
    weights: str
    epochs: int
    val_f1: Optional[float]
    val_acc: Optional[float]
    test_f1: Optional[float]
    test_acc: Optional[float]
    notes: str

    @property
    def group_label(self) -> str:
        return f"{self.model} ({self.weights})"


def _parse_float(value: str, field: str, run: str) -> Optional[float]:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        print(f"Warning: could not parse {field}='{value}' for run '{run}'.")
        return None


def _parse_int(value: str, field: str, run: str) -> Optional[int]:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        print(f"Warning: could not parse {field}='{value}' for run '{run}'.")
    return None


def _normalize_header(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _build_epoch_segments(
    points: List[Tuple[int, float]],
) -> List[Tuple[int, float, int, float]]:
    if not points:
        return []

    by_epoch: dict[int, List[float]] = {}
    for epoch, value in points:
        by_epoch.setdefault(epoch, []).append(value)

    epochs = sorted(by_epoch)
    if len(epochs) < 2:
        return []

    for values in by_epoch.values():
        values.sort()

    segments: List[Tuple[int, float, int, float]] = []
    for epoch, value in points:
        candidates = [candidate for candidate in epochs if candidate != epoch]
        if not candidates:
            continue

        min_dist = min(abs(candidate - epoch) for candidate in candidates)
        nearest_epochs = [candidate for candidate in candidates if abs(candidate - epoch) == min_dist]
        target_epoch = min(nearest_epochs)

        neighbor_values = by_epoch[target_epoch]
        neighbor_value = min(neighbor_values, key=lambda v: abs(v - value))
        segments.append((epoch, value, target_epoch, neighbor_value))

    return segments


def _load_val_acc(run: str, model: str, metrics_root: Path) -> Optional[float]:
    metrics_path = metrics_root / run / model / "metrics.json"
    if not metrics_path.exists():
        print(f"Warning: metrics.json not found for run '{run}' model '{model}'.")
        return None

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: metrics.json invalid for run '{run}' model '{model}'.")
        return None

    history = data.get("history")
    if not isinstance(history, list) or not history:
        print(f"Warning: metrics.json history missing for run '{run}' model '{model}'.")
        return None

    best_f1: Optional[float] = None
    best_acc: Optional[float] = None
    fallback_acc: Optional[float] = None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        f1_raw = entry.get("f1")
        acc_raw = entry.get("accuracy")
        f1 = None
        acc = None
        if f1_raw is not None:
            try:
                f1 = float(f1_raw)
            except (TypeError, ValueError):
                f1 = None
        if acc_raw is not None:
            try:
                acc = float(acc_raw)
            except (TypeError, ValueError):
                acc = None

        if f1 is not None and acc is not None:
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_acc = acc
        if f1 is None and acc is not None:
            if fallback_acc is None or acc > fallback_acc:
                fallback_acc = acc

    if best_acc is not None:
        return best_acc
    return fallback_acc


def parse_runs_table(summary_path: Path, *, metrics_root: Optional[Path] = None) -> List[RunRecord]:
    text = summary_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    table_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("| Run ") and "Epochs" in line:
            table_start = idx
            break

    if table_start is None:
        raise ValueError("Could not find the Runs (recent) table in summary.md.")

    header_cells = [_normalize_header(cell) for cell in lines[table_start].strip().strip("|").split("|")]
    header_map = {name: idx for idx, name in enumerate(header_cells) if name}

    def get_cell(cells: List[str], key: str) -> str:
        idx = header_map.get(key)
        if idx is None or idx >= len(cells):
            return ""
        return cells[idx]

    rows: List[RunRecord] = []
    for line in lines[table_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("##"):
            break
        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if get_cell(cells, "run").lower() == "run" or set(get_cell(cells, "run")) == {"-"}:
            continue
        if get_cell(cells, "run").startswith("---"):
            continue

        run = get_cell(cells, "run")
        model = get_cell(cells, "model")
        weights = get_cell(cells, "weights")
        epochs_raw = get_cell(cells, "epochs")
        val_f1_raw = get_cell(cells, "val_f1")
        val_acc_raw = get_cell(cells, "val_acc")
        test_f1_raw = get_cell(cells, "test_f1")
        test_acc_raw = get_cell(cells, "test_acc")
        notes = get_cell(cells, "notes")

        epochs = _parse_int(epochs_raw, "epochs", run)
        if epochs is None:
            continue
        if not run or not model or not weights:
            print(f"Warning: skipping row with missing run/model/weights: {cells}")
            continue

        val_acc = _parse_float(val_acc_raw, "val_acc", run)
        if val_acc is None and metrics_root is not None:
            val_acc = _load_val_acc(run, model, metrics_root)

        rows.append(
            RunRecord(
                run=run,
                model=model,
                weights=weights,
                epochs=epochs,
                val_f1=_parse_float(val_f1_raw, "val_f1", run),
                val_acc=val_acc,
                test_f1=_parse_float(test_f1_raw, "test_f1", run),
                test_acc=_parse_float(test_acc_raw, "test_acc", run),
                notes=notes,
            )
        )

    if not rows:
        raise ValueError("No run rows were parsed from summary.md.")
    return rows


def plot_metric(
    records: List[RunRecord],
    *,
    metric_name: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    groups = sorted({record.group_label for record in records})
    palette = {
        "dtd": "#1f77b4",
        "resnet18": "#2ca02c",
        "resnet50_imagenet": "#d62728",
        "squeezenet": "#f1c40f",
        "fallback": "#7f7f7f",
    }

    def _group_color(group: str) -> tuple[float, float, float]:
        normalized = group.strip().lower()
        if "resnet18" in normalized and "none" in normalized:
            base = palette["fallback"]
        elif "dtd" in normalized:
            base = palette["dtd"]
        elif "resnet18" in normalized:
            base = palette["resnet18"]
        elif "resnet50" in normalized and "imagenet" in normalized:
            base = palette["resnet50_imagenet"]
        elif "squeezenet" in normalized:
            base = palette["squeezenet"]
        else:
            base = palette["fallback"]
        # Keep points slightly pastel for readability while preserving category color.
        rgb = mcolors.to_rgb(base)
        pastel_mix = 0.2
        return tuple((1 - pastel_mix) * channel + pastel_mix * 1.0 for channel in rgb)

    fig, ax = plt.subplots(figsize=(9, 6))
    for group in groups:
        subset = [record for record in records if record.group_label == group]
        points = [
            (record.epochs, getattr(record, metric_name))
            for record in subset
            if getattr(record, metric_name) is not None
        ]
        points.sort(key=lambda pair: (pair[0], pair[1]))
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        if not x_vals:
            continue
        ax.scatter(
            x_vals,
            y_vals,
            label=group,
            color=_group_color(group),
            s=70,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

    xticks = sorted({record.epochs for record in records} | {0})
    if xticks:
        ax.set_xticks(xticks)

    ax.set_xlabel("Epochs")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Model (weights)", fontsize=9, title_fontsize=9, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate run summary charts from summary.md.")
    parser.add_argument("--summary-path", type=Path, default=Path("summary.md"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "run_summary_charts",
    )
    parser.add_argument(
        "--f1-metric",
        choices=["val_f1", "test_f1"],
        default="test_f1",
        help="Which F1 metric to plot.",
    )
    parser.add_argument(
        "--acc-metric",
        choices=["val_acc", "test_acc"],
        default="test_acc",
        help="Which accuracy metric to plot.",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root folder containing run outputs for val_acc lookup.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics_root = args.metrics_root if args.acc_metric == "val_acc" else None
    records = parse_runs_table(args.summary_path, metrics_root=metrics_root)

    f1_output = args.output_dir / f"run_summary_f1_vs_epochs_{args.f1_metric}.png"
    plot_metric(
        records,
        metric_name=args.f1_metric,
        output_path=f1_output,
        title=f"F1 vs Epochs ({args.f1_metric})",
        y_label="F1 score",
    )

    acc_output = args.output_dir / f"run_summary_accuracy_vs_epochs_{args.acc_metric}.png"
    plot_metric(
        records,
        metric_name=args.acc_metric,
        output_path=acc_output,
        title=f"Accuracy vs Epochs ({args.acc_metric})",
        y_label="Accuracy",
    )

    print(f"Saved charts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
