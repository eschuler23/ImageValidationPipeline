#!/usr/bin/env python3
"""
Plot ResNet50 DTD per-epoch validation curves from runresults.md.

This parses the "Resnet 50 DTD epoch documnetation" section in runresults.md,
extracts per-epoch val_f1/val_acc tables, and applies a small per-run epoch
offset (jitter) so overlapping runs remain visible. The Specs block is rendered
as a figure caption, and the legend labels are derived only from each run's
notes text. Two output files are generated: one for F1 and one for accuracy.

How to run:
  uv run python Image_Processing/Content_validation/runresults_resnet50_dtd_plot.py \
    --runresults-path runresults.md \
    --output-path reports/resnet50_dtd_epoch_curves_runresults.png

This will write:
  reports/resnet50_dtd_epoch_curves_runresults_f1.png
  reports/resnet50_dtd_epoch_curves_runresults_acc.png
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SECTION_HEADER_RE = re.compile(r"^#\s*Resnet\s+50\s+DTD\s+epoch", re.IGNORECASE)
RUN_HEADER_RE = re.compile(r"^#{3,6}\s*(\d+)\b")
TABLE_HEADER_RE = re.compile(r"^\s*Epoch\s*\|", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^\s*[-|\s]+$")
UNFREEZE_AFTER_RE = re.compile(r"unfreeze\s+after\s+(\d+)\s*epochs?", re.IGNORECASE)
FREEZE_EPOCHS_RE = re.compile(r"freeze_backbone_epochs\s*[:=]\s*(\d+)", re.IGNORECASE)
SPLIT_RE = re.compile(r"(val_split|test_split)\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
FROZEN_RE = re.compile(r"frozen[-\s]+backbone|backbone\s+frozen", re.IGNORECASE)


@dataclass(frozen=True)
class RunSeries:
    run_id: str
    epochs: List[int]
    f1: List[float]
    acc: List[float]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ResNet50 DTD epoch curves from runresults.md")
    parser.add_argument(
        "--runresults-path",
        type=Path,
        default=Path("runresults.md"),
        help="Path to runresults.md.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/resnet50_dtd_epoch_curves_runresults.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--epoch-offset",
        type=float,
        default=0.18,
        help="Constant per-run offset applied to epochs to reduce overlap.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Run root used for optional extra comparison runs.",
    )
    parser.add_argument(
        "--extra-run-names",
        nargs="*",
        default=["dtd_resnet50_seed21_ep15_freeze1_20260205"],
        help="Optional run folder names to append as extra comparison series.",
    )
    return parser.parse_args()


def _find_section(lines: List[str]) -> Tuple[int, int]:
    start = None
    for idx, line in enumerate(lines):
        if SECTION_HEADER_RE.match(line.strip()):
            start = idx
            break
    if start is None:
        raise ValueError("Could not find the Resnet 50 DTD section in runresults.md.")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("# ") and idx != start:
            end = idx
            break
    return start, end


def _collect_spec_items(lines: List[str]) -> List[str]:
    specs_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "specs":
            specs_idx = idx
            break
    if specs_idx is None:
        return []

    items: List[str] = []
    current: Optional[str] = None
    for line in lines[specs_idx + 1 :]:
        if RUN_HEADER_RE.match(line.strip()):
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            if current:
                items.append(current.strip())
            current = stripped.lstrip("-").strip()
        else:
            if current is None:
                current = stripped
            else:
                current = f"{current} {stripped}"

    if current:
        items.append(current.strip())

    return items


def _extract_specs(lines: List[str]) -> str:
    items = _collect_spec_items(lines)
    if not items:
        return "Specs summary: not found in runresults.md."

    summary = _summarize_specs(items)
    return f"Specs summary: {summary}"


def _summarize_specs(items: List[str]) -> str:
    kv: dict[str, str] = {}
    extras: List[str] = []

    for item in items:
        if ":" in item:
            key, value = item.split(":", 1)
            kv[key.strip().lower()] = value.strip()
        else:
            extras.append(item.strip())

    sentences: List[str] = []
    model = kv.get("model")
    weights = kv.get("weights")
    if model and weights:
        sentences.append(f"{model} initialized from {weights}.")
    elif model:
        sentences.append(f"{model} run.")
    elif weights:
        sentences.append(f"Initialized from {weights}.")

    training_bits: List[str] = []
    if kv.get("batch_size"):
        training_bits.append(f"batch size {kv['batch_size']}")
    if kv.get("lr"):
        training_bits.append(f"lr {kv['lr']}")
    if kv.get("unfreeze_lr"):
        training_bits.append(f"unfreeze lr {kv['unfreeze_lr']}")
    if kv.get("weight_decay"):
        training_bits.append(f"weight decay {kv['weight_decay']}")
    if kv.get("freeze_backbone_epochs"):
        training_bits.append(f"freeze backbone {kv['freeze_backbone_epochs']} epoch(s)")
    if kv.get("seed"):
        training_bits.append(f"seed {kv['seed']}")
    if training_bits:
        sentences.append(f"Training setup: {', '.join(training_bits)}.")

    if kv.get("split_strategy"):
        sentences.append(f"Split strategy: {kv['split_strategy']}.")
    if kv.get("preprocessing"):
        sentences.append(f"\nPreprocessing: {kv['preprocessing']}.")
    if kv.get("dataset"):
        sentences.append(f"Dataset: {kv['dataset']}.")

    for extra in extras:
        if not extra:
            continue
        if "f1" in extra.lower():
            sentences.append("Metric note: F1 is reported as a weighted score.")
        else:
            extra_clean = extra.rstrip(".")
            sentences.append(f"{extra_clean}.")

    if not sentences:
        return "details unavailable."
    return " ".join(sentences)


def _spec_defaults(items: List[str]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for item in items:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        if key == "freeze_backbone_epochs":
            try:
                defaults[key] = float(value.strip())
            except ValueError:
                continue
    return defaults


def _parse_table(rows: Iterable[str]) -> Tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    f1: List[float] = []
    acc: List[float] = []
    for row in rows:
        stripped = row.strip()
        if not stripped:
            break
        if TABLE_SEPARATOR_RE.match(stripped):
            continue
        if stripped.startswith("#"):
            break
        if "|" not in stripped:
            break

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        try:
            epoch = int(cells[0])
            val_f1 = float(cells[1])
            val_acc = float(cells[2])
        except ValueError:
            continue
        epochs.append(epoch)
        f1.append(val_f1)
        acc.append(val_acc)
    return epochs, f1, acc


def _format_split(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "?"


def _parse_unfreeze_epoch(block_lines: List[str], total_epochs: int) -> Optional[int]:
    for line in block_lines:
        match = UNFREEZE_AFTER_RE.search(line)
        if match:
            return int(match.group(1))
        match = FREEZE_EPOCHS_RE.search(line)
        if match:
            return int(match.group(1))
        if FROZEN_RE.search(line):
            return total_epochs
    return None


def _parse_splits(block_lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
    val_split = None
    test_split = None
    for line in block_lines:
        for match in SPLIT_RE.finditer(line):
            key = match.group(1).lower()
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            if key == "val_split":
                val_split = value
            elif key == "test_split":
                test_split = value
    return val_split, test_split


def _build_legend_label(
    block_lines: List[str],
    total_epochs: int,
    *,
    fallback_unfreeze: Optional[int],
    fallback_val_split: Optional[float],
    fallback_test_split: Optional[float],
) -> Tuple[str, Optional[int], Optional[float], Optional[float]]:
    unfreeze_epoch = _parse_unfreeze_epoch(block_lines, total_epochs)
    val_split, test_split = _parse_splits(block_lines)

    if unfreeze_epoch is None:
        unfreeze_epoch = fallback_unfreeze
    if val_split is None:
        val_split = fallback_val_split
    if test_split is None:
        test_split = fallback_test_split

    label = (
        f"epochs: {total_epochs}; "
        f"backbone unfreezes after {unfreeze_epoch if unfreeze_epoch is not None else '?'} epochs; "
        f"split: val={_format_split(val_split)}, test={_format_split(test_split)}"
    )
    return label, unfreeze_epoch, val_split, test_split


def parse_runs(lines: List[str], *, defaults: Optional[dict[str, float]] = None) -> List[RunSeries]:
    run_blocks: List[Tuple[str, List[str]]] = []
    current_id: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        match = RUN_HEADER_RE.match(line.strip())
        if match:
            if current_id is not None:
                run_blocks.append((current_id, current_lines))
            current_id = match.group(1)
            current_lines = [line]
        elif current_id is not None:
            current_lines.append(line)

    if current_id is not None:
        run_blocks.append((current_id, current_lines))

    runs: List[RunSeries] = []
    last_unfreeze: Optional[int] = None
    last_val_split: Optional[float] = None
    last_test_split: Optional[float] = None
    if defaults and "freeze_backbone_epochs" in defaults:
        last_unfreeze = int(defaults["freeze_backbone_epochs"])
    for run_id, block_lines in run_blocks:
        block_val_split, block_test_split = _parse_splits(block_lines)
        if block_val_split is not None:
            last_val_split = block_val_split
        if block_test_split is not None:
            last_test_split = block_test_split

        table_start = None
        for idx, line in enumerate(block_lines):
            if TABLE_HEADER_RE.match(line.strip()):
                table_start = idx
                break

        if table_start is None:
            continue

        epochs, f1, acc = _parse_table(block_lines[table_start + 1 :])
        if not epochs:
            continue

        label, last_unfreeze, last_val_split, last_test_split = _build_legend_label(
            block_lines,
            max(epochs),
            fallback_unfreeze=last_unfreeze,
            fallback_val_split=last_val_split,
            fallback_test_split=last_test_split,
        )
        runs.append(RunSeries(run_id=run_id, epochs=epochs, f1=f1, acc=acc, notes=label))

    return runs


def _to_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_from_preprocessing(preprocessing: dict) -> Tuple[Optional[float], Optional[float]]:
    splits = preprocessing.get("splits")
    if not isinstance(splits, dict):
        return None, None
    train = _to_int_or_none(splits.get("train_base"))
    val = _to_int_or_none(splits.get("val_base"))
    test = _to_int_or_none(splits.get("test_base"))
    if train is None or val is None or test is None:
        return None, None
    total = train + val + test
    if total <= 0:
        return None, None
    return val / total, test / total


def _load_run_folder_series(runs_root: Path, run_name: str) -> RunSeries:
    run_dir = runs_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metrics_path = run_dir / "resnet50" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("model") != "resnet50" or str(metrics.get("weights", "")).lower() != "dtd":
        raise ValueError(f"{run_name} is not a ResNet50 DTD run.")

    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"No history found in {metrics_path}")

    epochs: List[int] = []
    f1: List[float] = []
    acc: List[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        epoch_raw = row.get("epoch")
        f1_raw = row.get("f1")
        acc_raw = row.get("accuracy")
        if epoch_raw is None or f1_raw is None or acc_raw is None:
            continue
        epochs.append(int(epoch_raw))
        f1.append(float(f1_raw))
        acc.append(float(acc_raw))
    if not epochs:
        raise ValueError(f"No usable epoch rows in {metrics_path}")

    preprocessing_path = run_dir / "preprocessing.json"
    preprocessing = {}
    if preprocessing_path.exists():
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    training = preprocessing.get("training") if isinstance(preprocessing, dict) else {}
    training = training if isinstance(training, dict) else {}
    freeze = _to_int_or_none(training.get("freeze_backbone_epochs"))
    val_split, test_split = _split_from_preprocessing(preprocessing)

    notes = (
        f"{run_name}; epochs: {max(epochs)}; "
        f"backbone unfreezes after {freeze if freeze is not None else '?'} epochs; "
        f"split: val={_format_split(val_split)}, test={_format_split(test_split)}"
    )
    return RunSeries(run_id=run_name, epochs=epochs, f1=f1, acc=acc, notes=notes)


def _derive_output_paths(output_path: Path) -> Tuple[Path, Path]:
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "resnet50_dtd_epoch_curves_runresults"
    base = output_path.with_suffix("")
    parent = output_path.parent
    f1_path = parent / f"{base.name}_f1{suffix}"
    acc_path = parent / f"{base.name}_acc{suffix}"
    return f1_path, acc_path


def plot_metric(
    runs: List[RunSeries],
    caption: str,
    output_path: Path,
    epoch_offset: float,
    *,
    metric_name: str,
    y_label: str,
    title: str,
) -> None:
    if not runs:
        raise ValueError("No ResNet50 DTD runs with epoch tables were found.")

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6))

    colors = [
        "#1f77b4",
        "#4a90d9",
        "#0b5fa5",
        "#6ba6de",
        "#2f7fbe",
        "#8bbce5",
        "#165a8f",
    ]

    offsets: List[float] = []
    if len(runs) == 1:
        offsets = [0.0]
    else:
        spread = epoch_offset
        offsets = [
            -spread + (2 * spread) * idx / (len(runs) - 1) for idx in range(len(runs))
        ]

    max_epoch = max(max(run.epochs) for run in runs)
    xticks = list(range(1, max_epoch + 1))
    max_offset = max(abs(offset) for offset in offsets) if offsets else 0.0

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        offset = offsets[idx]
        x_vals = [epoch + offset for epoch in run.epochs]
        y_vals = run.f1 if metric_name == "f1" else run.acc

        ax.plot(x_vals, y_vals, marker="o", linewidth=1.7, color=color, label=run.notes)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(0.5 - max_offset, max_epoch + 0.5 + max_offset)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    legend_cols = 1 if len(labels) <= 3 else 2
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=8,
        frameon=False,
        title="Legend",
        title_fontsize=9,
        ncol=legend_cols,
    )

    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig.text(0.01, 0.02, caption, ha="left", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    text = args.runresults_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    section_start, section_end = _find_section(lines)
    section_lines = lines[section_start:section_end]

    caption = _extract_specs(section_lines)
    defaults = _spec_defaults(_collect_spec_items(section_lines))
    runs = parse_runs(section_lines, defaults=defaults)
    for run_name in args.extra_run_names:
        if not run_name:
            continue
        try:
            runs.append(_load_run_folder_series(args.runs_root, run_name))
        except Exception as exc:
            print(f"Skipping extra run '{run_name}': {exc}")

    f1_path, acc_path = _derive_output_paths(args.output_path)
    plot_metric(
        runs,
        caption,
        f1_path,
        args.epoch_offset,
        metric_name="f1",
        y_label="Val F1",
        title="ResNet50 DTD Validation F1 per Epoch",
    )
    plot_metric(
        runs,
        caption,
        acc_path,
        args.epoch_offset,
        metric_name="acc",
        y_label="Val Accuracy",
        title="ResNet50 DTD Validation Accuracy per Epoch",
    )
    print(f"Wrote: {f1_path}")
    print(f"Wrote: {acc_path}")
    print(f"Runs plotted: {len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
