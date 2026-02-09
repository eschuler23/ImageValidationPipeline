#!/usr/bin/env python3
"""
Evaluate trained runs on the seed-21 test split plus a new labeled folder.

This script:
1. Recovers the exact test split from each run's `dataset_manifest.csv`.
2. Verifies all requested runs share the same test split.
3. Adds labeled samples from a new image folder.
4. Runs inference for each model checkpoint.
5. Saves per-run:
   - combined predictions CSV
   - confusion metrics JSON (overall + per subset)
   - confusion-matrix panel PNG
   - FP/FN misclassification grid PNG with prediction probabilities

How to run:
  uv run python Image_Processing/Content_validation/evaluate_test_newonly_models.py \
    --run-names \
      dtd_resnet50_seed21_ep15_freeze5_val15_test5_20260205 \
      squeezenet1_1_imagenet_unfrozen_ep15_val15_test5_20260207 \
      resnet50_imagenet_ep15_freeze5_val15_test5_seed21 \
    --new-image-dir Imagedump_82_new_only \
    --new-labels-csv ground_truth_AURA82csv.csv \
    --new-label-column Quality \
    --new-positive-labels usable \
    --output-root reports/test40_newonly_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

import train_binary_classifier as tbc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EvalSample:
    subset: str
    image_path: Path
    filename: str
    project: str
    label_raw: str
    label_binary: int
    label_text: str


@dataclass(frozen=True)
class PredictionRecord:
    subset: str
    project: str
    filename: str
    image_path: str
    label_raw: str
    label_binary: int
    label_text: str
    pred_binary: int
    pred_text: str
    pred_score: float
    error_type: str


class EvalDataset(Dataset):
    def __init__(self, samples: Sequence[EvalSample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def normalize_label(value: str) -> str:
    return str(value).strip().lower()


def parse_label_list(values: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for value in values:
        for piece in str(value).split(","):
            cleaned = normalize_label(piece)
            if cleaned:
                labels.append(cleaned)
    return labels


def build_eval_transform(image_size: int):
    return tbc.build_transforms(
        image_size=image_size,
        random_augment=False,
        crop_jitter=False,
        jpeg_quality=None,
        jpeg_prob=0.0,
        noise_std=None,
        noise_prob=0.0,
        color_jitter=False,
        color_jitter_hue=0.0,
        color_jitter_saturation=0.0,
        color_jitter_contrast=0.0,
        brightness_jitter=False,
        brightness_jitter_delta=0.0,
        random_hflip=False,
        random_vflip=False,
        random_rotations=(),
        random_rotation_prob=0.0,
    )


def load_preprocessing(run_dir: Path) -> Tuple[int, Sequence[str], Sequence[str]]:
    preprocessing_path = run_dir / "preprocessing.json"
    if not preprocessing_path.exists():
        return 224, ["usable"], ["not usable"]
    payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    image_size = int(payload.get("image_size", 224))
    positive_labels = payload.get("positive_labels") or ["usable"]
    negative_labels = payload.get("negative_labels") or ["not usable"]
    return image_size, positive_labels, negative_labels


def resolve_model_dir(run_dir: Path, model_name: Optional[str]) -> Path:
    if model_name:
        model_dir = run_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir
    candidates = [path for path in run_dir.iterdir() if path.is_dir() and (path / "best_model.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No model directory with best_model.pt found under {run_dir}")
    if len(candidates) > 1:
        names = ", ".join(sorted(path.name for path in candidates))
        raise ValueError(f"Multiple model dirs found ({names}); pass --model-name.")
    return candidates[0]


def load_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[int, str], str, str]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {model_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = metrics.get("model", model_dir.name)
    weights = metrics.get("weights", "none")

    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {model_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)

    model = tbc.build_model(model_name, weights)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    run_dir = model_dir.parent
    _, positive_labels, negative_labels = load_preprocessing(run_dir)
    label_texts = tbc.build_label_texts(positive_labels, negative_labels)
    return model, label_texts, model_name, weights


def resolve_existing_path(path: Path, repo_root: Path) -> Path:
    if path.exists():
        return path
    candidate = (repo_root / path).resolve()
    if candidate.exists():
        return candidate
    return candidate


def load_test_split_samples(
    run_dir: Path,
    repo_root: Path,
    *,
    expected_count: int,
    positive_label_text: str,
    negative_label_text: str,
) -> List[EvalSample]:
    manifest_path = run_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset_manifest.csv in {run_dir}")

    rows: List[EvalSample] = []
    missing_images: List[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if normalize_label(row.get("split", "")) != "test":
                continue
            filename = str(row.get("filename", "")).strip()
            image_path_raw = str(row.get("image_path", "")).strip()
            if not filename or not image_path_raw:
                continue

            image_path = resolve_existing_path(Path(image_path_raw), repo_root)
            if not image_path.exists():
                missing_images.append(str(image_path))
                continue

            try:
                label_binary = int(float(row.get("label_binary", "0")))
            except ValueError:
                continue

            label_raw = str(row.get("label_raw", "")).strip()
            label_text = positive_label_text if label_binary == 1 else negative_label_text
            project = str(row.get("project_mapped", "") or row.get("project", "")).strip()
            rows.append(
                EvalSample(
                    subset="test_split",
                    image_path=image_path,
                    filename=filename,
                    project=project or "<missing>",
                    label_raw=label_raw,
                    label_binary=label_binary,
                    label_text=label_text,
                )
            )

    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} test images from manifest for run {run_dir.name}. "
            f"First missing path: {missing_images[0]}"
        )

    if expected_count > 0 and len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} test samples in {run_dir.name}, found {len(rows)}."
        )
    return rows


def load_new_samples(
    *,
    image_dir: Path,
    labels_csv: Path,
    filename_column: str,
    label_column: str,
    positive_labels: Sequence[str],
    negative_labels: Sequence[str],
    strict_labels: bool,
    project_name: str,
    positive_label_text: str,
    negative_label_text: str,
) -> Tuple[List[EvalSample], Dict[str, int]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"New image dir not found: {image_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"New labels csv not found: {labels_csv}")

    image_lookup: Dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_lookup[path.name] = path

    positive_set = {normalize_label(label) for label in positive_labels}
    negative_set = {normalize_label(label) for label in negative_labels}

    stats = {
        "rows": 0,
        "matched": 0,
        "missing_filename": 0,
        "missing_label": 0,
        "unknown_label": 0,
        "missing_image_file": 0,
        "duplicate_filename": 0,
    }
    seen_filenames: set[str] = set()
    samples: List[EvalSample] = []

    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if filename_column not in reader.fieldnames:
            raise ValueError(f"Column '{filename_column}' not found in {labels_csv}")
        if label_column not in reader.fieldnames:
            raise ValueError(f"Column '{label_column}' not found in {labels_csv}")

        for row in reader:
            stats["rows"] += 1
            filename = str(row.get(filename_column, "")).strip()
            label_raw = str(row.get(label_column, "")).strip()

            if not filename:
                stats["missing_filename"] += 1
                continue
            if not label_raw:
                stats["missing_label"] += 1
                continue
            if filename in seen_filenames:
                stats["duplicate_filename"] += 1
                continue
            seen_filenames.add(filename)

            normalized = normalize_label(label_raw)
            if normalized in positive_set:
                label_binary = 1
            elif normalized in negative_set:
                label_binary = 0
            elif strict_labels:
                stats["unknown_label"] += 1
                continue
            else:
                # For Quality-style CSVs, any non-positive annotation is negative.
                label_binary = 0

            image_path = image_lookup.get(filename)
            if image_path is None:
                stats["missing_image_file"] += 1
                continue

            label_text = positive_label_text if label_binary == 1 else negative_label_text
            samples.append(
                EvalSample(
                    subset="new_images",
                    image_path=image_path,
                    filename=filename,
                    project=project_name,
                    label_raw=label_raw,
                    label_binary=label_binary,
                    label_text=label_text,
                )
            )
            stats["matched"] += 1

    if not samples:
        raise ValueError("No new labeled samples were matched.")
    return samples, stats


def test_split_signature(samples: Sequence[EvalSample]) -> List[Tuple[str, int, str]]:
    return sorted((sample.filename, sample.label_binary, str(sample.image_path)) for sample in samples)


def infer_records(
    *,
    model: torch.nn.Module,
    label_texts: Dict[int, str],
    samples: Sequence[EvalSample],
    transform,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> List[PredictionRecord]:
    dataset = EvalDataset(samples, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records: List[PredictionRecord] = []
    with torch.no_grad():
        for images, sample_indices in loader:
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

            for sample_index, prob in zip(sample_indices.tolist(), probs):
                sample = samples[int(sample_index)]
                pred_binary = 1 if prob >= threshold else 0
                pred_text = label_texts.get(pred_binary, str(pred_binary))

                if sample.label_binary == 1 and pred_binary == 1:
                    error_type = "tp"
                elif sample.label_binary == 0 and pred_binary == 0:
                    error_type = "tn"
                elif sample.label_binary == 0 and pred_binary == 1:
                    error_type = "fp"
                else:
                    error_type = "fn"

                records.append(
                    PredictionRecord(
                        subset=sample.subset,
                        project=sample.project,
                        filename=sample.filename,
                        image_path=str(sample.image_path),
                        label_raw=sample.label_raw,
                        label_binary=sample.label_binary,
                        label_text=sample.label_text,
                        pred_binary=pred_binary,
                        pred_text=pred_text,
                        pred_score=float(prob),
                        error_type=error_type,
                    )
                )
    return records


def compute_metrics_for_records(records: Sequence[PredictionRecord]) -> Dict[str, float | int | List[List[int]]]:
    labels = [record.label_binary for record in records]
    preds = [record.pred_binary for record in records]
    metrics = tbc.compute_metrics(preds, labels)
    tn = int(round(metrics["tn"]))
    fp = int(round(metrics["fp"]))
    fn = int(round(metrics["fn"]))
    tp = int(round(metrics["tp"]))
    matrix = [[tn, fp], [fn, tp]]

    return {
        "count": len(records),
        "positive_count": int(sum(labels)),
        "negative_count": int(len(labels) - sum(labels)),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "matrix": matrix,
    }


def subset_records(records: Sequence[PredictionRecord], subset: str) -> List[PredictionRecord]:
    return [record for record in records if record.subset == subset]


def save_predictions_csv(records: Sequence[PredictionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset",
        "project",
        "filename",
        "image_path",
        "label_raw",
        "label_binary",
        "label_text",
        "pred_binary",
        "pred_text",
        "pred_score",
        "error_type",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["pred_score"] = f"{record.pred_score:.6f}"
            writer.writerow(row)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    text_w, text_h = _text_size(draw, text, font)
    left, top, right, bottom = box
    x = left + max(0, (right - left - text_w) // 2)
    y = top + max(0, (bottom - top - text_h) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def plot_confusion_panels(
    metrics_by_subset: Dict[str, Dict[str, float | int | List[List[int]]]],
    *,
    negative_text: str,
    positive_text: str,
    output_path: Path,
    title_prefix: str,
) -> None:
    scopes = [
        ("overall", "Overall"),
        ("test_split", "Test Split (40)"),
        ("new_images", "New Images"),
    ]
    margin = 20
    panel_w = 420
    panel_h = 280
    title_h = 48
    width = margin + len(scopes) * (panel_w + margin)
    height = title_h + panel_h + margin

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, 14), f"{title_prefix} - Confusion Matrices", fill=(0, 0, 0), font=font)

    for idx, (scope_key, scope_title) in enumerate(scopes):
        x0 = margin + idx * (panel_w + margin)
        y0 = title_h
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=2)
        draw.text((x0 + 10, y0 + 8), scope_title, fill=(0, 0, 0), font=font)

        metrics = metrics_by_subset.get(scope_key)
        if not metrics or int(metrics.get("count", 0)) == 0:
            draw.text((x0 + 10, y0 + 32), "no samples", fill=(80, 80, 80), font=font)
            continue

        matrix = metrics["matrix"]  # [[tn, fp], [fn, tp]]
        tn = int(matrix[0][0])
        fp = int(matrix[0][1])
        fn = int(matrix[1][0])
        tp = int(matrix[1][1])

        draw.text(
            (x0 + 10, y0 + 32),
            (
                f"count={int(metrics['count'])}  "
                f"acc={float(metrics['accuracy']):.3f}  f1={float(metrics['f1']):.3f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (x0 + 10, y0 + 50),
            (
                f"precision={float(metrics['precision']):.3f}  "
                f"recall={float(metrics['recall']):.3f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )

        table_left = x0 + 60
        table_top = y0 + 86
        cell_w = 140
        cell_h = 70

        # axis headers
        draw.text((table_left + 38, table_top - 18), f"Pred: {negative_text}", fill=(0, 0, 0), font=font)
        draw.text((table_left + cell_w + 26, table_top - 18), f"Pred: {positive_text}", fill=(0, 0, 0), font=font)
        draw.text((x0 + 8, table_top + 24), f"True: {negative_text}", fill=(0, 0, 0), font=font)
        draw.text((x0 + 8, table_top + cell_h + 24), f"True: {positive_text}", fill=(0, 0, 0), font=font)

        cells = [
            ((table_left, table_top, table_left + cell_w, table_top + cell_h), tn, "TN"),
            ((table_left + cell_w, table_top, table_left + 2 * cell_w, table_top + cell_h), fp, "FP"),
            ((table_left, table_top + cell_h, table_left + cell_w, table_top + 2 * cell_h), fn, "FN"),
            (
                (
                    table_left + cell_w,
                    table_top + cell_h,
                    table_left + 2 * cell_w,
                    table_top + 2 * cell_h,
                ),
                tp,
                "TP",
            ),
        ]
        for box, value, label in cells:
            color = (230, 244, 255)
            if label in {"FP", "FN"}:
                color = (255, 236, 236)
            draw.rectangle(box, outline=(0, 0, 0), fill=color, width=1)
            _draw_centered_text(draw, box, f"{label}\n{value}", font=font, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_misclassification_grid(
    records: Sequence[PredictionRecord],
    output_path: Path,
    *,
    columns: int,
    image_size: int,
    max_images: int,
) -> int:
    misclassified = [record for record in records if record.pred_binary != record.label_binary]
    misclassified.sort(key=lambda item: abs(item.pred_score - 0.5), reverse=True)
    if max_images > 0:
        misclassified = misclassified[:max_images]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    if not misclassified:
        canvas = Image.new("RGB", (700, 220), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 26), "No misclassifications", fill=(0, 0, 0), font=font)
        canvas.save(output_path)
        return 0

    total = len(misclassified)
    cols = max(1, columns)
    rows = (total + cols - 1) // cols

    padding = 12
    caption_h = 56
    tile_w = image_size
    tile_h = image_size + caption_h
    canvas_w = padding + cols * (tile_w + padding)
    canvas_h = padding + rows * (tile_h + padding)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, record in enumerate(misclassified):
        row = idx // cols
        col = idx % cols
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)

        image_path = Path(record.image_path)
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize((image_size, image_size))
        except OSError:
            image = Image.new("RGB", (image_size, image_size), (235, 235, 235))
            failed_draw = ImageDraw.Draw(image)
            failed_draw.text((8, 8), "image load failed", fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y))

        subset_text = "test" if record.subset == "test_split" else "new"
        border_color = (220, 70, 70) if record.error_type == "fp" else (220, 150, 50)
        draw.rectangle((x, y, x + tile_w, y + image_size), outline=border_color, width=3)

        title_lines = [
            _shorten(f"[{subset_text}] {record.label_text} -> {record.pred_text}", 42),
            f"p={record.pred_score:.3f}  {record.error_type.upper()}",
            _shorten(record.filename, 42),
        ]
        text_y = y + image_size + 2
        for line in title_lines:
            draw.text((x + 2, text_y), line, fill=(0, 0, 0), font=font)
            text_y += 16

    canvas.save(output_path)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate runs on test split + new-only labeled folder and export confusion + grids."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Directory containing run subfolders.",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        required=True,
        help="Run folder names to evaluate (under --runs-root).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Optional model subdir name override (if each run has multiple model folders).",
    )
    parser.add_argument(
        "--new-image-dir",
        type=Path,
        required=True,
        help="Folder containing new-only images.",
    )
    parser.add_argument(
        "--new-labels-csv",
        type=Path,
        required=True,
        help="CSV with labels for new-only images.",
    )
    parser.add_argument("--new-filename-column", type=str, default="filename")
    parser.add_argument("--new-label-column", type=str, default="Quality")
    parser.add_argument(
        "--new-positive-labels",
        nargs="+",
        default=["usable"],
        help="Label values to map to positive class.",
    )
    parser.add_argument(
        "--new-negative-labels",
        nargs="*",
        default=[],
        help="Optional explicit values to map to negative class.",
    )
    parser.add_argument(
        "--strict-new-labels",
        action="store_true",
        help="If set, skip new rows whose labels are not explicitly listed as positive/negative.",
    )
    parser.add_argument(
        "--new-project-name",
        type=str,
        default="AURA82_new_only",
        help="Project name written to prediction rows for new-only samples.",
    )
    parser.add_argument(
        "--expected-test-count",
        type=int,
        default=40,
        help="Expected test split size in each run manifest.",
    )
    parser.add_argument(
        "--reference-run-name",
        type=str,
        help=(
            "Run name whose manifest test split is used for all evaluations. "
            "Defaults to the first value in --run-names."
        ),
    )
    parser.add_argument(
        "--require-matching-test-splits",
        action="store_true",
        help=(
            "If set, fail when requested runs have different test splits. "
            "By default, a warning is shown and the reference split is used."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/test40_newonly_eval"),
        help="Output directory root (one folder per run).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4, help="Grid columns.")
    parser.add_argument(
        "--grid-max-images",
        type=int,
        default=0,
        help="Limit misclassification tiles (0 means all).",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Run inference + metrics only. Do not render PNG images.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root used to resolve relative image paths from manifests.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    runs_root = args.runs_root
    run_dirs = [runs_root / run_name for run_name in args.run_names]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_dirs_by_name = {run_dir.name: run_dir for run_dir in run_dirs}
    if args.reference_run_name:
        reference_run_dir = run_dirs_by_name.get(args.reference_run_name)
        if reference_run_dir is None:
            raise ValueError(
                f"reference run '{args.reference_run_name}' not found in --run-names."
            )
    else:
        reference_run_dir = run_dirs[0]

    # Use reference run labels as canonical display texts.
    _, ref_positive_labels, ref_negative_labels = load_preprocessing(reference_run_dir)
    positive_label_text = str(ref_positive_labels[0]) if ref_positive_labels else "usable"
    negative_label_text = str(ref_negative_labels[0]) if ref_negative_labels else "not usable"

    test_samples_by_run: Dict[str, List[EvalSample]] = {}
    for run_dir in run_dirs:
        samples = load_test_split_samples(
            run_dir,
            args.repo_root,
            expected_count=args.expected_test_count,
            positive_label_text=positive_label_text,
            negative_label_text=negative_label_text,
        )
        test_samples_by_run[run_dir.name] = samples

    reference_signature = test_split_signature(test_samples_by_run[reference_run_dir.name])
    mismatched_runs: List[str] = []
    for run_dir in run_dirs:
        if run_dir == reference_run_dir:
            continue
        signature = test_split_signature(test_samples_by_run[run_dir.name])
        if signature != reference_signature:
            mismatched_runs.append(run_dir.name)

    if mismatched_runs:
        mismatch_text = ", ".join(mismatched_runs)
        message = (
            "Requested runs do not share the same test split as reference run "
            f"'{reference_run_dir.name}'. Mismatched runs: {mismatch_text}"
        )
        if args.require_matching_test_splits:
            raise ValueError(message)
        print(f"WARNING: {message}")
        print(f"Continuing with reference split from '{reference_run_dir.name}'.")

    reference_test_samples = test_samples_by_run[reference_run_dir.name]
    new_positive_labels = parse_label_list(args.new_positive_labels)
    new_negative_labels = parse_label_list(args.new_negative_labels)
    new_samples, new_stats = load_new_samples(
        image_dir=args.new_image_dir,
        labels_csv=args.new_labels_csv,
        filename_column=args.new_filename_column,
        label_column=args.new_label_column,
        positive_labels=new_positive_labels,
        negative_labels=new_negative_labels,
        strict_labels=args.strict_new_labels,
        project_name=args.new_project_name,
        positive_label_text=positive_label_text,
        negative_label_text=negative_label_text,
    )

    combined_samples = list(reference_test_samples) + list(new_samples)

    device = tbc.resolve_device(args.device)
    print(
        f"Evaluating {len(run_dirs)} run(s) on {len(reference_test_samples)} test + "
        f"{len(new_samples)} new samples (total={len(combined_samples)}), device={device}."
    )
    print(f"Reference split run: {reference_run_dir.name}")
    print(f"New-label stats: {json.dumps(new_stats, ensure_ascii=False)}")

    for run_dir in run_dirs:
        model_dir = resolve_model_dir(run_dir, args.model_name)
        model, label_texts, model_name, weights = load_model(model_dir, device)
        image_size, _, _ = load_preprocessing(run_dir)
        transform = build_eval_transform(image_size)

        records = infer_records(
            model=model,
            label_texts=label_texts,
            samples=combined_samples,
            transform=transform,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=device,
        )

        metrics_by_subset: Dict[str, Dict[str, float | int | List[List[int]]]] = {
            "overall": compute_metrics_for_records(records),
            "test_split": compute_metrics_for_records(subset_records(records, "test_split")),
            "new_images": compute_metrics_for_records(subset_records(records, "new_images")),
        }

        run_output_dir = args.output_root / run_dir.name
        predictions_csv = run_output_dir / "combined_predictions.csv"
        metrics_json = run_output_dir / "metrics_summary.json"
        confusion_png = run_output_dir / "confusion_matrices.png"
        mis_grid_png = run_output_dir / "misclassification_grid.png"

        save_predictions_csv(records, predictions_csv)

        if args.skip_visuals:
            misclassified_count = sum(1 for record in records if record.pred_binary != record.label_binary)
        else:
            misclassified_count = render_misclassification_grid(
                records,
                mis_grid_png,
                columns=args.columns,
                image_size=image_size,
                max_images=args.grid_max_images,
            )
            plot_confusion_panels(
                metrics_by_subset,
                negative_text=negative_label_text,
                positive_text=positive_label_text,
                output_path=confusion_png,
                title_prefix=run_dir.name,
            )

        metrics_payload = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_dir": str(model_dir),
            "model": model_name,
            "weights": weights,
            "threshold": args.threshold,
            "device": str(device),
            "image_size": image_size,
            "counts": {
                "test_split": len(reference_test_samples),
                "new_images": len(new_samples),
                "combined": len(combined_samples),
                "misclassified": misclassified_count,
            },
            "reference_split_run": reference_run_dir.name,
            "mismatched_test_splits": mismatched_runs,
            "new_label_stats": new_stats,
            "metrics": metrics_by_subset,
            "visuals_generated": not args.skip_visuals,
            "artifacts": {
                "predictions_csv": str(predictions_csv),
                "metrics_json": str(metrics_json),
                "confusion_png": str(confusion_png) if not args.skip_visuals else None,
                "misclassification_grid_png": str(mis_grid_png) if not args.skip_visuals else None,
            },
        }
        run_output_dir.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        overall = metrics_by_subset["overall"]
        print(
            f"[{run_dir.name}] overall acc={float(overall['accuracy']):.3f}, "
            f"f1={float(overall['f1']):.3f}, "
            f"TN={int(overall['tn'])}, FP={int(overall['fp'])}, "
            f"FN={int(overall['fn'])}, TP={int(overall['tp'])}"
        )
        if args.skip_visuals:
            print(f"[{run_dir.name}] visuals skipped (--skip-visuals).")
        print(f"[{run_dir.name}] artifacts: {run_output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
