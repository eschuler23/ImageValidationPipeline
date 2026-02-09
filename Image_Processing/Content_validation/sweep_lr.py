#!/usr/bin/env python3
"""
Learning-rate sweep helper for the content-validation pipeline.

Why this exists
- The main pipeline (`main.py`) intentionally runs *one model per run* to avoid
  timeouts.
- When you want to compare learning rates (alpha values), repeating manual
  commands is tedious and error-prone.

What this script does
- Runs the same training pipeline once per (model, lr) pair.
- Calls `main.py` as a subprocess, so each run is isolated.
- Collects each run's `summary.json` and writes a consolidated Markdown report.

Usage (typical)
```
uv run python Image_Processing/Content_validation/sweep_lr.py \
  --models resnet50 \
  --lrs 1e-5 3e-5 1e-4 3e-4 \
  --run-name lr_sweep_resnet50 \
  -- \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --weights imagenet \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

Notes
- Do NOT pass `--lr`, `--models`, or `--run-name` in the training args; this
  script manages those.
- Pass any other training flags after `--` (or directly; they are forwarded).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, TypedDict


DEFAULT_RUN_ROOT = Path("Image_Processing/Content_validation/runs")
DEFAULT_REVIEW_ROOT = Path("Image_Processing/Content_validation/reviews")
SUMMARY_FILENAME = "summary.json"


class SummaryRow(TypedDict):
    model: str
    val_f1: float
    test_f1: float
    test_accuracy: float
    output_dir: str


class SweepRow(TypedDict):
    model: str
    lr: float
    val_f1: float | None
    test_f1: float | None
    test_accuracy: float | None
    run_name: str
    output_dir: str | None


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run a learning-rate sweep by calling main.py once per model + lr."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to sweep (each model is run separately).",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        required=True,
        help="Learning rates to try (e.g., 1e-5 3e-5 1e-4).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Base run name for grouping (defaults to timestamp).",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        help="Optional path for the consolidated review markdown.",
    )
    parser.add_argument(
        "--review-metric",
        choices=["val_f1", "test_f1", "test_accuracy"],
        default="val_f1",
        help="Metric used to highlight the best lr per model.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke main.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training.",
    )

    args, train_args = parser.parse_known_args()
    # Allow a `--` delimiter to cleanly separate sweep args from training args.
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    return args, train_args


def _find_flag_value(args: Iterable[str], flag: str) -> str | None:
    """Return the value for a `--flag value` or `--flag=value` entry."""

    args_list = list(args)
    for idx, item in enumerate(args_list):
        if item == flag:
            if idx + 1 >= len(args_list):
                raise ValueError(f"Expected a value after {flag}.")
            return args_list[idx + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _ensure_no_conflicts(train_args: Iterable[str]) -> None:
    """Disallow flags that this helper controls directly."""

    conflicts = ["--lr", "--models", "--run-name"]
    for flag in conflicts:
        if _find_flag_value(train_args, flag) is not None:
            raise ValueError(
                f"Do not pass {flag} in sweep_lr.py training args; "
                "this helper supplies it for each run."
            )
        if any(arg.startswith(f"{flag}=") for arg in train_args):
            raise ValueError(
                f"Do not pass {flag}=... in sweep_lr.py training args; "
                "this helper supplies it for each run."
            )


def _resolve_run_root(train_args: Iterable[str]) -> Path:
    run_root_value = _find_flag_value(train_args, "--run-root")
    if run_root_value:
        return Path(run_root_value)
    return DEFAULT_RUN_ROOT


def _format_lr_for_name(value: float) -> str:
    """Format lr for filenames while keeping it human-readable."""

    text = f"{value:g}"
    return text.replace(".", "p")


def _build_run_name(base: str, model: str, lr: float) -> str:
    lr_text = _format_lr_for_name(lr)
    return f"{base}_{model}_lr-{lr_text}"


def _run_training(
    *,
    python_exe: str,
    main_path: Path,
    model: str,
    lr: float,
    run_name: str,
    train_args: List[str],
    dry_run: bool,
) -> None:
    command = [
        python_exe,
        str(main_path),
        "--models",
        model,
        "--lr",
        str(lr),
        "--run-name",
        run_name,
        *train_args,
    ]
    print(f"Running: {shlex.join(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def _load_summary(summary_path: Path) -> SummaryRow:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary:
        raise ValueError(f"summary.json is empty: {summary_path}")
    return summary[0]


def _write_review(
    *,
    review_path: Path,
    rows: List[SweepRow],
    models: List[str],
    lrs: List[float],
    run_root: Path,
    review_metric: str,
    base_run_name: str,
) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)

    best_by_model: Dict[str, float] = {}
    for row in rows:
        model = row["model"]
        metric_value = row.get(review_metric)
        if metric_value is None:
            continue
        if model not in best_by_model or metric_value > best_by_model[model]:
            best_by_model[model] = metric_value

    lines: List[str] = []
    lines.append("# Learning-rate sweep review")
    lines.append("")
    lines.append(f"Base run name: `{base_run_name}`")
    lines.append(f"Run root: `{run_root}`")
    lines.append(f"Models: {', '.join(models)}")
    lines.append("Learning rates: " + ", ".join(f"{lr:g}" for lr in lrs))
    lines.append(f"Best metric per model: `{review_metric}`")
    lines.append("")
    lines.append(
        "| model | lr | val_f1 | test_f1 | test_acc | best | run_name | output_dir |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for row in rows:
        model = row["model"]
        lr_value = row["lr"]
        best_marker = ""
        metric_value = row.get(review_metric)
        if metric_value is not None and best_by_model.get(model) == metric_value:
            best_marker = "best"
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"{lr_value:g}",
                    f"{row.get('val_f1', float('nan')):.4f}",
                    f"{row.get('test_f1', float('nan')):.4f}",
                    f"{row.get('test_accuracy', float('nan')):.4f}",
                    best_marker,
                    row["run_name"],
                    row.get("output_dir") or "",
                ]
            )
            + " |"
        )

    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args, train_args = _parse_args()
    _ensure_no_conflicts(train_args)

    run_root = _resolve_run_root(train_args)
    base_run_name = args.run_name or time.strftime("lr_sweep_%Y%m%d_%H%M%S")
    review_path = args.review_file or (
        DEFAULT_REVIEW_ROOT / f"{base_run_name}_review.md"
    )

    main_path = Path(__file__).resolve().parent / "main.py"

    results: List[SweepRow] = []
    for model in args.models:
        for lr in args.lrs:
            run_name = _build_run_name(base_run_name, model, lr)
            _run_training(
                python_exe=args.python,
                main_path=main_path,
                model=model,
                lr=lr,
                run_name=run_name,
                train_args=train_args,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue
            summary_path = run_root / run_name / SUMMARY_FILENAME
            summary = _load_summary(summary_path)
            results.append(
                {
                    "model": summary.get("model", model),
                    "lr": lr,
                    "val_f1": summary.get("val_f1"),
                    "test_f1": summary.get("test_f1"),
                    "test_accuracy": summary.get("test_accuracy"),
                    "output_dir": summary.get("output_dir"),
                    "run_name": run_name,
                }
            )

    if args.dry_run:
        return

    _write_review(
        review_path=review_path,
        rows=results,
        models=args.models,
        lrs=args.lrs,
        run_root=run_root,
        review_metric=args.review_metric,
        base_run_name=base_run_name,
    )
    print(f"Review written to: {review_path}")


if __name__ == "__main__":
    main()
