"""Post-training evaluation artifacts for binary image classifiers."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


PredictionRecord = Dict[str, str]


def confusion_matrix(
    records: Sequence[PredictionRecord],
    labels: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    matrix = {label: {pred: 0 for pred in labels} for label in labels}
    for record in records:
        label = record.get("label_text", "")
        pred = record.get("pred_text", "")
        if label not in matrix:
            matrix[label] = {known: 0 for known in labels}
        if pred not in matrix[label]:
            matrix[label][pred] = 0
        matrix[label][pred] += 1
    return matrix


def binary_metrics(records: Sequence[PredictionRecord], positive_label: str) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    for record in records:
        label = record.get("label_text", "")
        pred = record.get("pred_text", "")
        if label == positive_label and pred == positive_label:
            tp += 1
        elif label != positive_label and pred != positive_label:
            tn += 1
        elif label != positive_label and pred == positive_label:
            fp += 1
        elif label == positive_label and pred != positive_label:
            fn += 1

    def div(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    total = tp + tn + fp + fn
    return {
        "accuracy": div(tp + tn, total),
        "sensitivity_recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def write_predictions_csv(path: Path, records: Sequence[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def misclassified(records: Sequence[PredictionRecord]) -> list[PredictionRecord]:
    return [
        record
        for record in records
        if record.get("label_text", "") != record.get("pred_text", "")
    ]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{font_name}", size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for word in text.split(" "):
        if len(word) > max_chars:
            pieces.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
        else:
            pieces.append(word)

    lines: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = piece
    if current:
        lines.append(current)
    return lines


def render_misclassified_grid(
    records: Sequence[PredictionRecord],
    output_path: Path,
    *,
    cols: int = 3,
    thumb_w: int = 280,
    thumb_h: int = 220,
    label_h: int = 96,
) -> Optional[Path]:
    failures = misclassified(records)
    if not failures:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(failures) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    font_bold = _font(13, bold=True)

    for index, record in enumerate(failures):
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        image_path = Path(record.get("image_path", ""))

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                paste_x = x + (thumb_w - image.width) // 2
                paste_y = y + (thumb_h - image.height) // 2
                canvas.paste(image, (paste_x, paste_y))
        except OSError as exc:
            draw.text((x + 8, y + 80), f"Could not load image: {exc}", fill="red", font=font)

        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h + label_h - 1], outline=(205, 205, 205))
        text_y = y + thumb_h + 7
        header = (
            f"true: {record.get('label_text', '')}  "
            f"pred: {record.get('pred_text', '')}  "
            f"score: {record.get('pred_score', '')}"
        )
        draw.text((x + 8, text_y), header, fill=(0, 0, 0), font=font_bold)
        text_y += 19
        for line in _wrap_text(record.get("filename", ""), 36)[:4]:
            draw.text((x + 8, text_y), line, fill=(30, 30, 30), font=font)
            text_y += 15

    canvas.save(output_path)
    return output_path


def _format_matrix(matrix: Mapping[str, Mapping[str, int]], labels: Sequence[str]) -> str:
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    divider = "|---|" + "|".join("---" for _ in labels) + "|"
    rows = [header, divider]
    for label in labels:
        rows.append("| " + label + " | " + " | ".join(str(matrix.get(label, {}).get(pred, 0)) for pred in labels) + " |")
    return "\n".join(rows)


def _format_config(config: Mapping[str, object]) -> list[str]:
    keys = [
        "model",
        "weights",
        "device",
        "csv_path",
        "image_root",
        "image_dir",
        "label_column",
        "positive_labels",
        "negative_labels",
        "epochs",
        "batch_size",
        "lr",
        "val_split",
        "test_split",
        "split_strategy",
        "augment",
    ]
    lines: list[str] = []
    for key in keys:
        if key in config:
            lines.append(f"- `{key}`: `{config[key]}`")
    return lines


def write_eval_report(
    *,
    output_dir: Path,
    config: Mapping[str, object],
    labels: Sequence[str],
    positive_label: str,
    best_epoch: Optional[int],
    best_records: Sequence[PredictionRecord],
    final_records: Sequence[PredictionRecord],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_matrix = confusion_matrix(best_records, labels)
    final_matrix = confusion_matrix(final_records, labels)
    best_metrics = binary_metrics(best_records, positive_label)
    final_metrics = binary_metrics(final_records, positive_label)

    best_predictions_path = output_dir / "best_val_predictions.csv"
    final_predictions_path = output_dir / "final_val_predictions.csv"
    write_predictions_csv(best_predictions_path, best_records)
    write_predictions_csv(final_predictions_path, final_records)

    best_grid = render_misclassified_grid(best_records, output_dir / "best_misclassified_grid.png")
    final_grid = render_misclassified_grid(final_records, output_dir / "final_misclassified_grid.png")

    payload: Dict[str, object] = {
        "title": "Eval Script",
        "config": dict(config),
        "best_epoch": best_epoch,
        "labels": list(labels),
        "positive_label": positive_label,
        "best": {
            "confusion_matrix": best_matrix,
            "metrics": best_metrics,
            "misclassified_count": len(misclassified(best_records)),
            "predictions_csv": str(best_predictions_path),
            "misclassified_grid": str(best_grid) if best_grid else None,
            "misclassified_filenames": [record.get("filename", "") for record in misclassified(best_records)],
        },
        "final": {
            "confusion_matrix": final_matrix,
            "metrics": final_metrics,
            "misclassified_count": len(misclassified(final_records)),
            "predictions_csv": str(final_predictions_path),
            "misclassified_grid": str(final_grid) if final_grid else None,
            "misclassified_filenames": [record.get("filename", "") for record in misclassified(final_records)],
        },
    }

    with (output_dir / "eval_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    report_lines = [
        "# Eval Script",
        "",
        "## 1. Configurations",
        *_format_config(config),
        "",
        "## 2. Best Epoch",
        f"- Best epoch: `{best_epoch}`",
        "",
        "## 3. Confusion Matrix",
        "### Best Epoch",
        _format_matrix(best_matrix, labels),
        "",
        "### Final Model",
        _format_matrix(final_matrix, labels),
        "",
        "## 4. Falsely Classified Images",
        "### Best Epoch Filenames",
        *[f"- `{record.get('filename', '')}` true=`{record.get('label_text', '')}` pred=`{record.get('pred_text', '')}` score=`{record.get('pred_score', '')}`" for record in misclassified(best_records)],
        "",
        "### Final Model Filenames",
        *[f"- `{record.get('filename', '')}` true=`{record.get('label_text', '')}` pred=`{record.get('pred_text', '')}` score=`{record.get('pred_score', '')}`" for record in misclassified(final_records)],
    ]
    if best_grid:
        report_lines.extend(["", f"Best epoch grid: `{best_grid}`"])
    if final_grid:
        report_lines.extend(["", f"Final model grid: `{final_grid}`"])

    with (output_dir / "eval_report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    return payload
