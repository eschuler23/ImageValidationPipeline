#!/usr/bin/env python3
"""
Binary content-validation training.

This module contains the core training logic for the content-validation
pipeline. The new `main.py` entry point orchestrates the pipeline by calling
small, documented step functions. Those step functions still rely on the
utilities in this file (data loading, augmentation expansion, model building,
training loops, and reporting helpers).

Keep this file as the source of truth for the training mechanics; the pipeline
entry point should only wire these pieces together.

How to run (multi-project CSV + Images/ root):
  uv run python Image_Processing/Content_validation/train_binary_classifier.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --project-column project \
    --filename-column filename \
    --label-column "usability considering nfp" \
    --positive-labels "usable" \
    --negative-labels "not usable" \
    --models resnet18 resnet50 squeezenet1_1 \
    --weights imagenet \
    --device auto \
    --decode-percent-newlines \
    --save-val-grid

Optional DTD init (requires a local checkpoint):
  uv run python Image_Processing/Content_validation/train_binary_classifier.py \
    --csv-path ground_truth.csv \
    --image-root Images \
    --project-column project \
    --filename-column filename \
    --label-column "usability considering nfp" \
    --positive-labels "usable" \
    --negative-labels "not usable" \
    --models resnet50 \
    --weights dtd \
    --init-checkpoint checkpoints/resnet50_dtd.pt \
    --device auto \
    --save-val-grid
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import time
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score


@dataclass
class Sample:
    image_path: Path
    label: int
    label_raw: str
    filename: str
    filename_raw: str
    project: Optional[str]
    project_mapped: Optional[str]


@dataclass(frozen=True)
class AugmentationSettings:
    blur_keep: Tuple[float, ...]
    blur_flip: Tuple[float, ...]
    blur_only_positive: bool
    blur_size_aware: bool
    blur_size_small_max_mp: float
    blur_size_medium_max_mp: float
    blur_switch_small: float
    blur_switch_medium: float
    blur_switch_large: float
    balance_target: Optional[float]
    balance_seed: int


@dataclass(frozen=True)
class RandomBlurConfig:
    """Configuration for on-the-fly blur with optional label flipping."""

    enabled: bool
    probability: float
    candidates: Tuple[float, ...]
    range_min: Optional[float]
    range_max: Optional[float]
    only_positive: bool
    size_aware: bool
    size_small_max_mp: float
    size_medium_max_mp: float
    switch_small: float
    switch_medium: float
    switch_large: float
    reject_radius: float


@dataclass
class SampleVariant:
    sample: Sample
    augmentation: str
    label: int
    label_source: str
    blur_radius: Optional[float] = None


Batch = Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]


class SummaryRow(TypedDict):
    model: str
    val_f1: float
    test_accuracy: float
    test_f1: float
    output_dir: str


def normalize_label(label: str) -> str:
    return label.strip().lower()


def parse_label_list(values: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for value in values:
        for piece in value.split(","):
            cleaned = normalize_label(piece)
            if cleaned:
                labels.append(cleaned)
    return labels


def load_project_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Project map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Project map JSON must be a mapping of CSV project -> folder name.")
    return {str(key): str(value) for key, value in payload.items()}


def _decode_percent_newlines(value: str, enabled: bool) -> str:
    if not enabled or not value:
        return value
    return value.replace("%0A", "\n").replace("%0a", "\n")


def build_label_texts(positive_labels: Sequence[str], negative_labels: Sequence[str]) -> Dict[int, str]:
    positive = positive_labels[0] if positive_labels else "positive"
    negative = negative_labels[0] if negative_labels else "negative"
    return {1: positive, 0: negative}


def parse_rotations(values: Sequence[int]) -> Tuple[int, ...]:
    rotations: List[int] = []
    for value in values:
        if value not in (90, 180, 270):
            raise ValueError("Rotations must be 90, 180, or 270 degrees.")
        rotations.append(value)
    return tuple(rotations)


def parse_float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def expand_range(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) != 3:
        raise ValueError("Range values must be: min max step")
    start, end, step = (float(values[0]), float(values[1]), float(values[2]))
    if step <= 0:
        raise ValueError("Range step must be > 0")
    radii: List[float] = []
    current = start
    while current <= end + 1e-9:
        radii.append(round(current, 6))
        current += step
    return radii


def pixel_count_from_path(image_path: Path) -> Optional[int]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError:
        return None
    return width * height


def resolve_blur_switch_threshold_pixels(
    pixel_count: Optional[int],
    *,
    size_small_max_mp: float,
    size_medium_max_mp: float,
    switch_small: float,
    switch_medium: float,
    switch_large: float,
    fallback: float,
) -> float:
    if pixel_count is None:
        return fallback
    size_mp = pixel_count / 1_000_000
    if size_mp <= size_small_max_mp:
        return switch_small
    if size_mp <= size_medium_max_mp:
        return switch_medium
    return switch_large


def resolve_blur_switch_threshold(image_path: Path, augment: AugmentationSettings) -> float:
    return resolve_blur_switch_threshold_pixels(
        pixel_count_from_path(image_path),
        size_small_max_mp=augment.blur_size_small_max_mp,
        size_medium_max_mp=augment.blur_size_medium_max_mp,
        switch_small=augment.blur_switch_small,
        switch_medium=augment.blur_switch_medium,
        switch_large=augment.blur_switch_large,
        fallback=augment.blur_switch_medium,
    )


def base_variants(samples: Sequence[Sample]) -> List[SampleVariant]:
    return [
        SampleVariant(
            sample=sample,
            augmentation="none",
            label=sample.label,
            label_source="csv",
        )
        for sample in samples
    ]


def build_train_variants(samples: Sequence[Sample], augment: AugmentationSettings) -> List[SampleVariant]:
    base_list = base_variants(samples)
    keep_variants: List[SampleVariant] = []
    flip_variants: List[SampleVariant] = []

    for sample in samples:
        blur_switch_threshold: Optional[float] = None
        if augment.blur_size_aware and (augment.blur_keep or augment.blur_flip):
            blur_switch_threshold = resolve_blur_switch_threshold(sample.image_path, augment)
        for radius in augment.blur_keep:
            if blur_switch_threshold is not None and radius > blur_switch_threshold:
                continue
            keep_variants.append(
                SampleVariant(
                    sample=sample,
                    augmentation=f"blur_keep_r{radius:g}",
                    label=sample.label,
                    label_source="blur_keep",
                    blur_radius=radius,
                )
            )
        for radius in augment.blur_flip:
            if blur_switch_threshold is not None and radius <= blur_switch_threshold:
                continue
            if augment.blur_only_positive and sample.label != 1:
                continue
            flip_variants.append(
                SampleVariant(
                    sample=sample,
                    augmentation=f"blur_flip_r{radius:g}",
                    label=0,
                    label_source="blur_flip",
                    blur_radius=radius,
                )
            )

    if augment.balance_target is None:
        return base_list + keep_variants + flip_variants

    baseline_variants = base_list + keep_variants
    positives = sum(1 for variant in baseline_variants if variant.label == 1)
    negatives = len(baseline_variants) - positives
    desired_negatives = int(round(positives * (1.0 - augment.balance_target) / augment.balance_target))
    needed = desired_negatives - negatives

    if needed <= 0:
        print(
            "Balance target reached without blur-flip variants: "
            f"pos={positives} neg={negatives} target={augment.balance_target:.2f}"
        )
        return baseline_variants

    if not flip_variants:
        print(
            "Balance target requested but no blur-flip variants available: "
            f"pos={positives} neg={negatives} target={augment.balance_target:.2f}"
        )
        return baseline_variants

    if needed >= len(flip_variants):
        print(
            "Balance target exceeds blur-flip candidates; using all blur-flip variants: "
            f"pos={positives} neg={negatives} needed={needed} candidates={len(flip_variants)}"
        )
        return baseline_variants + flip_variants

    rng = random.Random(augment.balance_seed)
    selected = rng.sample(flip_variants, needed)
    final_negatives = negatives + len(selected)
    print(
        "Balanced train variants using blur-flip sampling: "
        f"pos={positives} neg={final_negatives} "
        f"selected={len(selected)} candidates={len(flip_variants)} "
        f"target={augment.balance_target:.2f}"
    )
    return baseline_variants + selected


def apply_augmentation(image: Image.Image, variant: SampleVariant) -> Image.Image:
    if variant.blur_radius is not None:
        image = image.filter(ImageFilter.GaussianBlur(radius=variant.blur_radius))

    return image


def unpack_batch(batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, str]]]:
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError("Unexpected batch structure from DataLoader.")


def load_samples(
    csv_path: Path,
    image_dir: Optional[Path],
    image_root: Optional[Path],
    label_column: str,
    positive_labels: Sequence[str],
    negative_labels: Sequence[str],
    filename_column: str = "filename",
    project_column: str = "project",
    project_map: Optional[Dict[str, str]] = None,
    decode_percent_newlines: bool = False,
    dedupe_filenames: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[List[Sample], Dict[str, int]]:
    positive_set = {normalize_label(label) for label in positive_labels}
    negative_set = {normalize_label(label) for label in negative_labels}
    project_map = project_map or {}
    stats = {
        "rows": 0,
        "matched": 0,
        "missing_label": 0,
        "missing_filename": 0,
        "missing_project": 0,
        "unknown_label": 0,
        "missing_file": 0,
        "duplicate_filename": 0,
        "decoded_percent_newlines": 0,
    }
    samples: List[Sample] = []
    seen_filenames: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["rows"] += 1
            label_raw = row.get(label_column, "")
            filename_raw = row.get(filename_column, "")
            filename_norm = _decode_percent_newlines(filename_raw, decode_percent_newlines)
            if filename_norm != filename_raw:
                stats["decoded_percent_newlines"] += 1
            project_raw = row.get(project_column, "") if image_root else ""

            if not label_raw:
                stats["missing_label"] += 1
                continue
            if not filename_raw:
                stats["missing_filename"] += 1
                continue
            if image_root and not project_raw:
                stats["missing_project"] += 1
                continue

            normalized = normalize_label(label_raw)
            if normalized in positive_set:
                label = 1
            elif normalized in negative_set:
                label = 0
            else:
                stats["unknown_label"] += 1
                continue

            if dedupe_filenames:
                if filename_norm in seen_filenames:
                    stats["duplicate_filename"] += 1
                    continue
                seen_filenames.add(filename_norm)

            project_mapped: str | None = None
            if image_root:
                project_mapped = project_map.get(project_raw, project_raw)
                image_path = image_root / project_mapped / filename_norm
            else:
                image_path = image_dir / filename_norm if image_dir else Path(filename_norm)

            if not image_path.exists():
                stats["missing_file"] += 1
                continue

            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    label_raw=label_raw,
                    filename=filename_norm,
                    filename_raw=filename_raw,
                    project=project_raw or None,
                    project_mapped=project_mapped,
                )
            )
            stats["matched"] += 1
            if max_samples and len(samples) >= max_samples:
                break

    return samples, stats


def split_indices(total: int, val_split: float, test_split: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_count = int(total * test_split)
    val_count = int(total * val_split)

    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]
    return train_idx, val_idx, test_idx


def split_indices_stratified(
    labels: Sequence[int],
    *,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    groups: Dict[int, List[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in sorted(groups.keys()):
        indices = groups[label]
        rng.shuffle(indices)
        total = len(indices)

        test_count = int(round(total * test_split))
        val_count = int(round(total * val_split))

        if total >= 3:
            if test_count == 0:
                test_count = 1
            if val_count == 0:
                val_count = 1
        elif total == 2 and val_count == 0 and test_count == 0:
            val_count = 1

        while test_count + val_count >= total and (test_count > 0 or val_count > 0):
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1

        test_idx.extend(indices[:test_count])
        val_idx.extend(indices[test_count : test_count + val_count])
        train_idx.extend(indices[test_count + val_count :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


class RandomJpegCompression:
    def __init__(self, quality_range: Optional[Tuple[int, int]], probability: float):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.quality_range or random.random() > self.probability:
            return image
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class RandomGaussianNoise:
    def __init__(self, std: Optional[float], probability: float):
        self.std = std
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std is None or random.random() > self.probability:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)


def build_transforms(
    image_size: int,
    random_augment: bool,
    crop_jitter: bool,
    jpeg_quality: Optional[Tuple[int, int]],
    jpeg_prob: float,
    noise_std: Optional[float],
    noise_prob: float,
    color_jitter: bool,
    color_jitter_hue: float,
    color_jitter_saturation: float,
    color_jitter_contrast: float,
    brightness_jitter: bool,
    brightness_jitter_delta: float,
    random_hflip: bool,
    random_vflip: bool,
    random_rotations: Sequence[int],
    random_rotation_prob: float,
) -> transforms.Compose:
    transform_list: List[transforms.Transform] = []

    if random_augment:
        transform_list.append(
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        )
    elif crop_jitter:
        transform_list.append(
            transforms.RandomResizedCrop(image_size, scale=(0.95, 1.0), ratio=(0.98, 1.02))
        )
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    if random_hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if random_vflip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if random_rotations:
        rotation_transforms = [
            transforms.Lambda(lambda img, angle=angle: img.rotate(angle)) for angle in random_rotations
        ]
        rotation_choice = transforms.RandomChoice(rotation_transforms)
        if random_rotation_prob < 1.0:
            transform_list.append(transforms.RandomApply([rotation_choice], p=random_rotation_prob))
        else:
            transform_list.append(rotation_choice)

    if color_jitter or brightness_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness_jitter_delta if brightness_jitter else 0.0,
                contrast=color_jitter_contrast if color_jitter else 0.0,
                saturation=color_jitter_saturation if color_jitter else 0.0,
                hue=color_jitter_hue if color_jitter else 0.0,
            )
        )

    if jpeg_quality:
        transform_list.append(RandomJpegCompression(jpeg_quality, jpeg_prob))

    transform_list.append(transforms.ToTensor())
    if noise_std is not None:
        transform_list.append(RandomGaussianNoise(noise_std, noise_prob))

    transform_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(transform_list)


def sample_random_blur_radius(
    random_blur: RandomBlurConfig,
    *,
    threshold: Optional[float],
) -> Optional[float]:
    if random_blur.candidates:
        candidates = list(random_blur.candidates)
        if threshold is not None and random_blur.reject_radius > 0:
            candidates = [radius for radius in candidates if abs(radius - threshold) > random_blur.reject_radius]
        if not candidates:
            raise ValueError(
                "Random blur candidates all fall within the reject radius. "
                "Adjust the candidate list or reject radius."
            )
        return random.choice(candidates)

    if random_blur.range_min is None or random_blur.range_max is None:
        return None

    if threshold is None or random_blur.reject_radius <= 0:
        return random.uniform(random_blur.range_min, random_blur.range_max)

    reject_min = threshold - random_blur.reject_radius
    reject_max = threshold + random_blur.reject_radius
    segments: List[Tuple[float, float]] = []
    if random_blur.range_min < reject_min:
        segments.append((random_blur.range_min, min(reject_min, random_blur.range_max)))
    if random_blur.range_max > reject_max:
        segments.append((max(reject_max, random_blur.range_min), random_blur.range_max))

    segments = [(low, high) for low, high in segments if high > low]
    if not segments:
        raise ValueError(
            "Random blur range falls entirely inside the reject radius window. "
            "Adjust the range or reject radius."
        )

    total = sum(high - low for low, high in segments)
    pick = random.random() * total
    for low, high in segments:
        span = high - low
        if pick <= span:
            return random.uniform(low, high)
        pick -= span
    return random.uniform(segments[-1][0], segments[-1][1])


class CsvImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleVariant],
        transform: transforms.Compose,
        return_meta: bool = False,
        random_blur: Optional[RandomBlurConfig] = None,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.return_meta = return_meta
        self.random_blur = random_blur

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Batch:
        variant = self.samples[index]
        sample = variant.sample
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
        image = apply_augmentation(image, variant)
        label_value = variant.label
        label_source = variant.label_source
        blur_radius = None

        if self.random_blur and self.random_blur.enabled:
            if self.random_blur.only_positive and label_value != 1:
                pass
            elif random.random() <= self.random_blur.probability:
                threshold = None
                if self.random_blur.size_aware:
                    pixel_count = image.size[0] * image.size[1]
                    threshold = resolve_blur_switch_threshold_pixels(
                        pixel_count,
                        size_small_max_mp=self.random_blur.size_small_max_mp,
                        size_medium_max_mp=self.random_blur.size_medium_max_mp,
                        switch_small=self.random_blur.switch_small,
                        switch_medium=self.random_blur.switch_medium,
                        switch_large=self.random_blur.switch_large,
                        fallback=self.random_blur.switch_medium,
                    )
                blur_radius = sample_random_blur_radius(self.random_blur, threshold=threshold)
                if blur_radius is not None:
                    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    label_source = "random_blur_keep"
                    if threshold is not None and blur_radius > threshold:
                        label_value = 0
                        label_source = "random_blur_flip"
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        if not self.return_meta:
            return image_tensor, label_tensor

        meta = {
            "image_path": str(sample.image_path),
            "filename": sample.filename,
            "filename_raw": sample.filename_raw,
            "project": sample.project or "",
            "project_mapped": sample.project_mapped or "",
            "augmentation": variant.augmentation,
            "label_raw": sample.label_raw,
            "label_source": label_source,
            "random_blur_radius": "" if blur_radius is None else f"{blur_radius:.4f}",
        }
        return image_tensor, label_tensor, meta


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name: str, weights: str) -> nn.Module:
    use_imagenet = weights == "imagenet"

    if model_name == "resnet18":
        model = _build_resnet(models.resnet18, "ResNet18_Weights", use_imagenet)
        # Binary logit head (no output activation).
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "resnet50":
        model = _build_resnet(models.resnet50, "ResNet50_Weights", use_imagenet)
        # Binary logit head (no output activation).
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "squeezenet1_1":
        model = _build_squeezenet(models.squeezenet1_1, "SqueezeNet1_1_Weights", use_imagenet)
        # Binary logit head: remove the final ReLU so BCEWithLogitsLoss receives
        # unconstrained logits (negative and positive values).
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        model.num_classes = 1
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def _build_resnet(builder, weights_name: str, use_imagenet: bool) -> nn.Module:
    if use_imagenet:
        try:
            weights_enum = getattr(models, weights_name).DEFAULT
            return builder(weights=weights_enum)
        except AttributeError:
            return builder(pretrained=True)
    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)


def _build_squeezenet(builder, weights_name: str, use_imagenet: bool) -> nn.Module:
    if use_imagenet:
        try:
            weights_enum = getattr(models, weights_name).DEFAULT
            return builder(weights=weights_enum)
        except AttributeError:
            return builder(pretrained=True)
    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_last_block(model: nn.Module) -> None:
    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1)
    probs = torch.sigmoid(logits)
    return (probs >= 0.5).long()


def compute_metrics(preds: Iterable[int], labels: Iterable[int]) -> Dict[str, float]:
    preds_list = list(preds)
    labels_list = list(labels)
    total = len(labels_list)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tp = sum(1 for p, y in zip(preds_list, labels_list) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds_list, labels_list) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds_list, labels_list) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds_list, labels_list) if p == 0 and y == 1)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = float(f1_score(labels_list, preds_list, average="weighted", zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits_to_predictions(logits)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
    return all_preds, all_labels


def collect_predictions_with_meta(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int], List[float], List[Dict[str, str]]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_meta: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch in loader:
            images, labels, meta = unpack_batch(batch)
            if meta is None:
                raise ValueError("Validation loader is missing metadata for grid output.")
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            batch_size = len(preds)
            for idx in range(batch_size):
                all_meta.append({key: value[idx] for key, value in meta.items()})

    return all_preds, all_labels, all_probs, all_meta


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    preds, labels = collect_predictions(model, loader, device)
    return compute_metrics(preds, labels)


def save_validation_predictions(
    output_dir: Path,
    preds: Sequence[int],
    labels: Sequence[int],
    probs: Sequence[float],
    meta: Sequence[Dict[str, str]],
    label_texts: Dict[int, str],
) -> Tuple[Path, List[Dict[str, str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "val_predictions.csv"
    rows: List[Dict[str, str]] = []

    for pred, label, prob, row_meta in zip(preds, labels, probs, meta):
        rows.append(
            {
                "project": row_meta.get("project", ""),
                "project_mapped": row_meta.get("project_mapped", ""),
                "filename": row_meta.get("filename", ""),
                "filename_raw": row_meta.get("filename_raw", ""),
                "image_path": row_meta.get("image_path", ""),
                "augmentation": row_meta.get("augmentation", ""),
                "label_raw": row_meta.get("label_raw", ""),
                "label_source": row_meta.get("label_source", ""),
                "label_binary": str(label),
                "label_text": label_texts.get(label, str(label)),
                "pred_binary": str(pred),
                "pred_text": label_texts.get(pred, str(pred)),
                "pred_score": f"{prob:.6f}",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path, rows


def save_validation_grids(
    output_dir: Path,
    records: Sequence[Dict[str, str]],
    *,
    grid_cols: int,
    max_images_per_grid: Optional[int],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for --save-val-grid.") from exc

    if not records:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    if max_images_per_grid is None or max_images_per_grid <= 0:
        max_images_per_grid = len(records)

    saved_paths: List[Path] = []
    for start in range(0, len(records), max_images_per_grid):
        chunk = records[start : start + max_images_per_grid]
        if not chunk:
            continue

        cols = max(1, grid_cols)
        rows = int(math.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))

        if rows == 1 and cols == 1:
            axes_list = [axes]
        elif rows == 1 or cols == 1:
            axes_list = list(axes)
        else:
            axes_list = [ax for row in axes for ax in row]

        for ax in axes_list[len(chunk) :]:
            ax.axis("off")

        for ax, record in zip(axes_list, chunk):
            image_path = record.get("image_path", "")
            label_raw = record.get("label_raw", "")
            pred_text = record.get("pred_text", "")
            pred_score = record.get("pred_score", "")
            title = f"gt: {label_raw}\\npred: {pred_text} ({pred_score})"

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                ax.imshow(image)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")

            ax.set_title(title, fontsize=7)
            ax.axis("off")

        grid_path = output_dir / f"val_grid_{start // max_images_per_grid + 1:02d}.png"
        fig.tight_layout()
        fig.savefig(str(grid_path), dpi=150)
        plt.close(fig)
        saved_paths.append(grid_path)

    return saved_paths


def write_dataset_manifest(
    output_path: Path,
    *,
    splits: Dict[str, Sequence[SampleVariant]],
    label_texts: Dict[int, str],
) -> None:
    rows: List[Dict[str, str]] = []
    for split_name, variants in splits.items():
        for variant in variants:
            sample = variant.sample
            rows.append(
                {
                    "split": split_name,
                    "project": sample.project or "",
                    "project_mapped": sample.project_mapped or "",
                    "filename": sample.filename,
                    "filename_raw": sample.filename_raw,
                    "image_path": str(sample.image_path),
                    "label_raw": sample.label_raw,
                    "label_binary": str(variant.label),
                    "label_text": label_texts.get(variant.label, str(variant.label)),
                    "label_source": variant.label_source,
                    "augmentation": variant.augmentation,
                    "blur_radius": "" if variant.blur_radius is None else str(variant.blur_radius),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_label_map(csv_path: Path, label_column: str, filename_column: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get(filename_column, "")
            label_raw = row.get(label_column, "")
            if not filename or label_raw is None:
                continue
            mapping[filename] = label_raw
    return mapping


def filter_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, int],
    List[str],
    List[Tuple[str, torch.Size, torch.Size]],
]:
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    stats = {"kept": 0, "dropped_missing": 0, "dropped_shape": 0}
    dropped_missing: List[str] = []
    dropped_shape: List[Tuple[str, torch.Size, torch.Size]] = []

    for key, value in state_dict.items():
        if key not in model_state:
            stats["dropped_missing"] += 1
            dropped_missing.append(key)
            continue
        if model_state[key].shape != value.shape:
            stats["dropped_shape"] += 1
            dropped_shape.append((key, model_state[key].shape, value.shape))
            continue
        filtered[key] = value
        stats["kept"] += 1

    return filtered, stats, dropped_missing, dropped_shape


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    train_variants: Sequence[SampleVariant],
    val_variants: Sequence[SampleVariant],
    test_variants: Sequence[SampleVariant],
    output_dir: Path,
    label_texts: Dict[int, str],
) -> SummaryRow:
    train_transform = build_transforms(
        image_size=args.image_size,
        random_augment=args.augment,
        crop_jitter=args.augment_crop_jitter,
        jpeg_quality=args.augment_jpeg_quality,
        jpeg_prob=args.augment_jpeg_prob,
        noise_std=args.augment_noise_std,
        noise_prob=args.augment_noise_prob,
        color_jitter=args.augment_color_jitter,
        color_jitter_hue=args.augment_color_jitter_hue,
        color_jitter_saturation=args.augment_color_jitter_saturation,
        color_jitter_contrast=args.augment_color_jitter_contrast,
        brightness_jitter=args.augment_brightness_jitter,
        brightness_jitter_delta=args.augment_brightness_jitter_delta,
        random_hflip=args.augment_random_hflip,
        random_vflip=args.augment_random_vflip,
        random_rotations=args.augment_random_rotations,
        random_rotation_prob=args.augment_random_rotation_prob,
    )
    eval_transform = build_transforms(
        image_size=args.image_size,
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

    random_blur = RandomBlurConfig(
        enabled=args.augment_random_blur,
        probability=args.augment_random_blur_prob,
        candidates=tuple(getattr(args, "augment_random_blur_candidates", ())),
        range_min=args.augment_random_blur_range[0] if args.augment_random_blur_range else None,
        range_max=args.augment_random_blur_range[1] if args.augment_random_blur_range else None,
        only_positive=args.augment_blur_only_positive,
        size_aware=args.augment_blur_size_aware,
        size_small_max_mp=args.blur_size_small_max_mp,
        size_medium_max_mp=args.blur_size_medium_max_mp,
        switch_small=args.blur_switch_small,
        switch_medium=args.blur_switch_medium,
        switch_large=args.blur_switch_large,
        reject_radius=args.augment_random_blur_reject_radius,
    )

    train_dataset = CsvImageDataset(train_variants, train_transform, random_blur=random_blur)
    val_dataset = CsvImageDataset(val_variants, eval_transform, return_meta=args.save_val_grid)
    test_dataset = CsvImageDataset(test_variants, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    model = build_model(model_name, args.weights)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        filtered_state, stats, dropped_missing, dropped_shape = filter_state_dict(
            model, state_dict
        )
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(
            f"Loaded checkpoint: kept={stats['kept']} dropped_missing={stats['dropped_missing']} "
            f"dropped_shape={stats['dropped_shape']} missing={len(missing)} unexpected={len(unexpected)}"
        )
        if args.checkpoint_verbose:
            if dropped_missing:
                print("Checkpoint dropped_missing (not in model):")
                for key in dropped_missing:
                    print(f"  - {key}")
            if dropped_shape:
                print("Checkpoint dropped_shape (shape mismatch):")
                for key, model_shape, ckpt_shape in dropped_shape:
                    print(f"  - {key}: model={list(model_shape)} checkpoint={list(ckpt_shape)}")
            if missing:
                print("Checkpoint missing (expected by model):")
                for key in missing:
                    print(f"  - {key}")
            if unexpected:
                print("Checkpoint unexpected (not used by model):")
                for key in unexpected:
                    print(f"  - {key}")

    model = model.to(device)

    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model)

    if args.freeze_batchnorm:
        freeze_batchnorm(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    pos_weight = None
    if args.use_class_weights:
        positives = sum(variant.label for variant in train_variants)
        negatives = len(train_variants) - positives
        if positives > 0:
            pos_weight = torch.tensor([negatives / positives], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print(f"AMP requested but not available on device '{device.type}'.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    grad_accum_steps = max(1, args.grad_accum_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_batchnorm:
            freeze_batchnorm(model)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            with autocast_ctx:
                logits = model(images).view(-1)
                loss = criterion(logits, labels)
                scaled_loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += loss.item() * images.size(0)

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
            if args.unfreeze_last_block:
                unfreeze_last_block(model)
            else:
                unfreeze_all(model)
            if args.freeze_batchnorm:
                freeze_batchnorm(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.unfreeze_lr,
                weight_decay=args.weight_decay,
            )

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1,
            }

        print(
            f"[{model_name}] Epoch {epoch:02d} "
            f"loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, output_dir / "best_model.pt")

    if best_state:
        model.load_state_dict(best_state["model_state"])

    test_metrics = evaluate(model, test_loader, device)

    val_predictions_path: Optional[Path] = None
    val_grid_paths: List[Path] = []
    if args.save_val_grid:
        preds, labels, probs, meta = collect_predictions_with_meta(model, val_loader, device)
        val_predictions_path, records = save_validation_predictions(
            output_dir=output_dir,
            preds=preds,
            labels=labels,
            probs=probs,
            meta=meta,
            label_texts=label_texts,
        )
        val_grid_paths = save_validation_grids(
            output_dir=output_dir,
            records=records,
            grid_cols=args.val_grid_cols,
            max_images_per_grid=args.val_grid_max_images,
        )

    metrics_payload = {
        "model": model_name,
        "weights": args.weights,
        "device": str(device),
        "train_samples": len(train_variants),
        "val_samples": len(val_variants),
        "test_samples": len(test_variants),
        "history": history,
        "test_metrics": test_metrics,
        "val_predictions_csv": str(val_predictions_path) if val_predictions_path else None,
        "val_grids": [str(path) for path in val_grid_paths],
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    return {
        "model": model_name,
        "val_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "output_dir": str(output_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a binary content-validation classifier.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth.csv"),
        help="Path to the CSV with labels (default: ground_truth.csv).",
    )
    parser.add_argument("--image-dir", type=Path, help="Directory containing images (single-folder mode).")
    parser.add_argument("--image-root", type=Path, help="Root containing project subfolders (multi-project mode).")
    parser.add_argument("--label-column", type=str, required=True, help="Column name for labels.")
    parser.add_argument(
        "--positive-labels",
        nargs="+",
        required=True,
        help="Labels treated as positive. Separate multiple labels with spaces or commas.",
    )
    parser.add_argument(
        "--negative-labels",
        nargs="+",
        required=True,
        help="Labels treated as negative. Separate multiple labels with spaces or commas.",
    )
    parser.add_argument("--filename-column", type=str, default="filename", help="CSV column containing file names.")
    parser.add_argument("--project-column", type=str, default="project", help="CSV column containing project name.")
    parser.add_argument("--project-map", type=Path, help="Optional JSON mapping of CSV project -> folder name.")
    parser.add_argument(
        "--decode-percent-newlines",
        action="store_true",
        help="Decode %0A in CSV filenames into literal newlines before matching.",
    )
    parser.add_argument(
        "--dedupe-filenames",
        action="store_true",
        default=True,
        help="Drop duplicate filenames (keep first occurrence).",
    )
    parser.add_argument(
        "--no-dedupe-filenames",
        dest="dedupe_filenames",
        action="store_false",
        help="Keep duplicate filenames.",
    )

    parser.add_argument("--models", nargs="+", default=["resnet18"], help="Models to train.")
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none", "dtd"],
        default="imagenet",
        help="Weight initialization (use dtd with --init-checkpoint).",
    )
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint to load (strict=False).")
    parser.add_argument(
        "--checkpoint-verbose",
        action="store_true",
        help="Print detailed checkpoint key differences (missing, unexpected, shape mismatches).",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = batch-size * steps).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument(
        "--unfreeze-last-block",
        action="store_true",
        help="Unfreeze only the last backbone block (e.g., ResNet layer4) after the freeze period.",
    )
    parser.add_argument(
        "--freeze-batchnorm",
        action="store_true",
        help="Freeze BatchNorm layers (no running stats updates or gradients).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision when supported by the device.",
    )
    parser.add_argument("--augment", action="store_true", help="Enable light random augmentation.")
    parser.add_argument("--augment-crop-jitter", action="store_true", help="Enable small crop/resize jitter.")
    parser.add_argument(
        "--augment-random-hflip",
        action="store_true",
        default=None,
        help="Random horizontal flip on the fly (defaults to on when --augment is set).",
    )
    parser.add_argument(
        "--augment-random-vflip",
        action="store_true",
        default=None,
        help="Random vertical flip on the fly.",
    )
    parser.add_argument(
        "--augment-random-rotations",
        nargs="+",
        type=int,
        default=[],
        help="Random on-the-fly rotations (90, 180, 270).",
    )
    parser.add_argument(
        "--augment-random-rotation-prob",
        type=float,
        default=0.5,
        help="Probability to apply a random rotation when rotations are enabled.",
    )
    parser.add_argument(
        "--augment-random-blur",
        action="store_true",
        help="Apply random blur on the fly (labels may flip based on size thresholds).",
    )
    parser.add_argument(
        "--augment-random-blur-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Uniform random blur radius range (overrides blur keep/flip ranges when set).",
    )
    parser.add_argument(
        "--augment-random-blur-prob",
        type=float,
        default=0.3,
        help="Probability of applying random blur per sample.",
    )
    parser.add_argument(
        "--augment-random-blur-reject-radius",
        type=float,
        default=4.0,
        help="Reject radius around blur flip thresholds when sampling random blur.",
    )
    parser.add_argument(
        "--augment-jpeg-quality",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Random JPEG compression quality range (e.g., 60 95).",
    )
    parser.add_argument("--augment-jpeg-prob", type=float, default=0.3, help="JPEG compression probability.")
    parser.add_argument("--augment-noise-std", type=float, help="Gaussian noise std (0-1 scale).")
    parser.add_argument("--augment-noise-prob", type=float, default=0.3, help="Gaussian noise probability.")
    parser.add_argument(
        "--augment-color-jitter",
        action="store_true",
        help="Enable color jitter (hue/saturation/contrast; off by default).",
    )
    parser.add_argument(
        "--augment-color-jitter-hue",
        type=float,
        default=0.02,
        help="Hue jitter range for color jitter (0-0.5).",
    )
    parser.add_argument(
        "--augment-color-jitter-saturation",
        type=float,
        default=0.2,
        help="Saturation jitter range for color jitter.",
    )
    parser.add_argument(
        "--augment-color-jitter-contrast",
        type=float,
        default=0.1,
        help="Contrast jitter range for color jitter.",
    )
    parser.add_argument(
        "--augment-brightness-jitter",
        action="store_true",
        help="Enable brightness jitter (off by default).",
    )
    parser.add_argument(
        "--augment-brightness-jitter-delta",
        type=float,
        default=0.1,
        help="Brightness jitter range when enabled.",
    )
    parser.add_argument(
        "--augment-blur-keep",
        nargs="+",
        type=float,
        default=[],
        help="Blur radii to apply while keeping the original label.",
    )
    parser.add_argument(
        "--augment-blur-flip",
        nargs="+",
        type=float,
        default=[],
        help="Blur radii that flip the label to negative.",
    )
    parser.add_argument(
        "--augment-blur-keep-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        help="Generate keep-label blur radii from a range.",
    )
    parser.add_argument(
        "--augment-blur-flip-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        help="Generate flip-label blur radii from a range.",
    )
    parser.add_argument(
        "--augment-blur-only-positive",
        action="store_true",
        help="Only blur positive-label samples.",
    )
    parser.add_argument(
        "--augment-blur-size-aware",
        action="store_true",
        help="Use pixel-size thresholds (MP) to decide when blur flips labels.",
    )
    parser.add_argument(
        "--blur-size-small-max-mp",
        type=float,
        default=2.0,
        help="Max image size in megapixels treated as 'small' for blur label switching.",
    )
    parser.add_argument(
        "--blur-size-medium-max-mp",
        type=float,
        default=10.0,
        help="Max image size in megapixels treated as 'medium' for blur label switching.",
    )
    parser.add_argument(
        "--blur-switch-small",
        type=float,
        default=2.0,
        help="Max blur radius before label flip for small images.",
    )
    parser.add_argument(
        "--blur-switch-medium",
        type=float,
        default=10.0,
        help="Max blur radius before label flip for medium images.",
    )
    parser.add_argument(
        "--blur-switch-large",
        type=float,
        default=20.0,
        help="Max blur radius before label flip for very large images.",
    )
    parser.add_argument(
        "--balance-train-target",
        type=float,
        help="Target positive fraction for train variants after blur-flip augmentation (e.g., 0.5).",
    )
    parser.add_argument(
        "--balance-train-seed",
        type=int,
        default=42,
        help="Seed for sampling blur-flip variants when balancing.",
    )
    parser.add_argument("--use-class-weights", action="store_true", help="Use class weighting for imbalance.")

    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Split strategy for train/val/test (stratified preserves label distribution).",
    )
    parser.add_argument(
        "--stratify-columns",
        nargs="+",
        help=(
            "CSV columns to stratify on (defaults to label + Blurr when available "
            "in the main pipeline)."
        ),
    )
    parser.add_argument(
        "--stratify-report-columns",
        nargs="+",
        help="CSV columns to report split balance for (defaults to all label-like columns).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, help="Optional cap for quicker tests.")
    parser.add_argument("--save-val-grid", action="store_true", help="Save validation grid and predictions CSV.")
    parser.add_argument("--val-grid-cols", type=int, default=4)
    parser.add_argument("--val-grid-max-images", type=int, default=0)

    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("Image_Processing/Content_validation/runs"),
        help="Root output directory for runs.",
    )
    parser.add_argument("--run-name", type=str, help="Optional run name. Defaults to timestamp.")

    return parser


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize and validate CLI arguments for the training pipeline."""
    args.positive_labels = parse_label_list(args.positive_labels)
    args.negative_labels = parse_label_list(args.negative_labels)

    if not args.positive_labels or not args.negative_labels:
        raise ValueError("Both positive and negative label lists are required.")

    if (args.image_dir is None) == (args.image_root is None):
        raise ValueError("Provide exactly one of --image-dir or --image-root.")

    if args.weights == "dtd" and not args.init_checkpoint:
        raise ValueError("DTD weights require --init-checkpoint.")

    args.augment_random_rotations = parse_rotations(args.augment_random_rotations)
    if args.augment_random_hflip is None:
        args.augment_random_hflip = bool(args.augment)
    if args.augment_random_vflip is None:
        args.augment_random_vflip = False
    if not (0.0 <= args.augment_random_rotation_prob <= 1.0):
        raise ValueError("--augment-random-rotation-prob must be in [0, 1].")
    if args.val_grid_max_images is not None and args.val_grid_max_images <= 0:
        args.val_grid_max_images = None

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")

    blur_keep = parse_float_list(args.augment_blur_keep)
    blur_keep.extend(expand_range(args.augment_blur_keep_range or []))
    blur_flip = parse_float_list(args.augment_blur_flip)
    blur_flip.extend(expand_range(args.augment_blur_flip_range or []))

    blur_candidates = tuple(sorted(set(blur_keep + blur_flip)))

    if args.augment_random_blur:
        if not (0.0 <= args.augment_random_blur_prob <= 1.0):
            raise ValueError("--augment-random-blur-prob must be in [0, 1].")
        if args.augment_random_blur_range is not None:
            if len(args.augment_random_blur_range) != 2:
                raise ValueError("--augment-random-blur-range requires MIN MAX.")
            blur_min, blur_max = args.augment_random_blur_range
            if blur_min < 0 or blur_max < 0:
                raise ValueError("--augment-random-blur-range values must be >= 0.")
            if blur_min > blur_max:
                raise ValueError("--augment-random-blur-range MIN must be <= MAX.")
            args.augment_random_blur_range = (float(blur_min), float(blur_max))
            args.augment_random_blur_candidates = ()
        else:
            if not blur_candidates:
                raise ValueError(
                    "Random blur enabled but no candidates provided. "
                    "Pass --augment-random-blur-range or a blur keep/flip range."
                )
            args.augment_random_blur_candidates = blur_candidates
            args.augment_random_blur_range = None

        # Disable deterministic blur variants when random blur is enabled.
        args.augment_blur_keep = ()
        args.augment_blur_flip = ()
    else:
        if args.augment_random_blur_range is not None:
            raise ValueError("--augment-random-blur-range requires --augment-random-blur.")
        args.augment_random_blur_candidates = ()
        args.augment_blur_keep = tuple(sorted(set(blur_keep)))
        args.augment_blur_flip = tuple(sorted(set(blur_flip)))

    if args.augment_jpeg_quality is not None:
        if len(args.augment_jpeg_quality) != 2:
            raise ValueError("--augment-jpeg-quality requires MIN MAX.")
        if args.augment_jpeg_quality[0] > args.augment_jpeg_quality[1]:
            raise ValueError("--augment-jpeg-quality MIN must be <= MAX.")
        args.augment_jpeg_quality = (
            int(args.augment_jpeg_quality[0]),
            int(args.augment_jpeg_quality[1]),
        )
    if not (0.0 <= args.augment_jpeg_prob <= 1.0):
        raise ValueError("--augment-jpeg-prob must be in [0, 1].")

    if args.augment_noise_std is not None and args.augment_noise_std < 0:
        raise ValueError("--augment-noise-std must be >= 0.")
    if not (0.0 <= args.augment_noise_prob <= 1.0):
        raise ValueError("--augment-noise-prob must be in [0, 1].")
    if args.augment_random_blur_reject_radius < 0:
        raise ValueError("--augment-random-blur-reject-radius must be >= 0.")
    if args.augment_color_jitter_hue < 0 or args.augment_color_jitter_hue > 0.5:
        raise ValueError("--augment-color-jitter-hue must be in [0, 0.5].")
    if args.augment_color_jitter_saturation < 0:
        raise ValueError("--augment-color-jitter-saturation must be >= 0.")
    if args.augment_color_jitter_contrast < 0:
        raise ValueError("--augment-color-jitter-contrast must be >= 0.")
    if args.augment_brightness_jitter_delta < 0:
        raise ValueError("--augment-brightness-jitter-delta must be >= 0.")
    if args.blur_size_small_max_mp <= 0 or args.blur_size_medium_max_mp <= 0:
        raise ValueError("Blur size thresholds must be positive MP values.")
    if not (args.blur_size_small_max_mp < args.blur_size_medium_max_mp):
        raise ValueError("Blur size MP thresholds must be strictly increasing.")
    if (
        args.blur_switch_small < 0
        or args.blur_switch_medium < 0
        or args.blur_switch_large < 0
    ):
        raise ValueError("Blur switch radii must be >= 0.")
    if args.balance_train_target is not None:
        if not (0.0 < args.balance_train_target < 1.0):
            raise ValueError("--balance-train-target must be in (0, 1).")

    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    args = prepare_args(args)

    set_seed(args.seed)

    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.image_dir and not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")
    if args.image_root and not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    project_map = load_project_map(args.project_map)

    samples, stats = load_samples(
        csv_path=csv_path,
        image_dir=args.image_dir,
        image_root=args.image_root,
        label_column=args.label_column,
        positive_labels=args.positive_labels,
        negative_labels=args.negative_labels,
        filename_column=args.filename_column,
        project_column=args.project_column,
        project_map=project_map,
        decode_percent_newlines=args.decode_percent_newlines,
        dedupe_filenames=args.dedupe_filenames,
        max_samples=args.max_samples,
    )

    if not samples:
        raise ValueError("No samples matched the provided label mapping.")

    if args.split_strategy == "stratified":
        labels = [sample.label for sample in samples]
        train_idx, val_idx, test_idx = split_indices_stratified(
            labels,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )
    else:
        train_idx, val_idx, test_idx = split_indices(
            len(samples), args.val_split, args.test_split, args.seed
        )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    augment_settings = AugmentationSettings(
        blur_keep=args.augment_blur_keep,
        blur_flip=args.augment_blur_flip,
        blur_only_positive=args.augment_blur_only_positive,
        blur_size_aware=args.augment_blur_size_aware,
        blur_size_small_max_mp=args.blur_size_small_max_mp,
        blur_size_medium_max_mp=args.blur_size_medium_max_mp,
        blur_switch_small=args.blur_switch_small,
        blur_switch_medium=args.blur_switch_medium,
        blur_switch_large=args.blur_switch_large,
        balance_target=args.balance_train_target,
        balance_seed=args.balance_train_seed,
    )

    train_variants = build_train_variants(train_samples, augment_settings)
    val_variants = base_variants(val_samples)
    test_variants = base_variants(test_samples)

    label_texts = build_label_texts(args.positive_labels, args.negative_labels)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = args.run_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    stats_payload: Dict[str, object] = {
        **stats,
        "samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_variants": len(train_variants),
        "val_variants": len(val_variants),
        "test_variants": len(test_variants),
        "train_augmentation_counts": dict(
            Counter(v.augmentation for v in train_variants)
        ),
    }

    with (run_root / "data_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)

    preprocessing_payload = {
        "csv_path": str(csv_path),
        "image_dir": str(args.image_dir) if args.image_dir else None,
        "image_root": str(args.image_root) if args.image_root else None,
        "project_column": args.project_column if args.image_root else None,
        "filename_column": args.filename_column,
        "label_column": args.label_column,
        "positive_labels": args.positive_labels,
        "negative_labels": args.negative_labels,
        "project_map": str(args.project_map) if args.project_map else None,
        "decode_percent_newlines": args.decode_percent_newlines,
        "dedupe_filenames": args.dedupe_filenames,
        "random_augmentation": args.augment,
        "augmentation": {
            "random_hflip": args.augment_random_hflip,
            "random_vflip": args.augment_random_vflip,
            "random_rotations": list(args.augment_random_rotations),
            "random_rotation_prob": args.augment_random_rotation_prob,
            "random_blur": args.augment_random_blur,
            "random_blur_prob": args.augment_random_blur_prob,
            "random_blur_candidates": list(getattr(args, "augment_random_blur_candidates", ())),
            "random_blur_range": (
                list(args.augment_random_blur_range)
                if args.augment_random_blur_range is not None
                else None
            ),
            "random_blur_reject_radius": args.augment_random_blur_reject_radius,
            "blur_keep": list(args.augment_blur_keep),
            "blur_flip": list(args.augment_blur_flip),
            "blur_only_positive": args.augment_blur_only_positive,
            "blur_size_aware": args.augment_blur_size_aware,
            "blur_size_small_max_mp": args.blur_size_small_max_mp,
            "blur_size_medium_max_mp": args.blur_size_medium_max_mp,
            "blur_switch_small": args.blur_switch_small,
            "blur_switch_medium": args.blur_switch_medium,
            "blur_switch_large": args.blur_switch_large,
            "balance_train_target": args.balance_train_target,
            "balance_train_seed": args.balance_train_seed,
            "crop_jitter": args.augment_crop_jitter,
            "jpeg_quality": args.augment_jpeg_quality,
            "jpeg_prob": args.augment_jpeg_prob,
            "noise_std": args.augment_noise_std,
            "noise_prob": args.augment_noise_prob,
            "color_jitter": args.augment_color_jitter,
            "color_jitter_hue": args.augment_color_jitter_hue,
            "color_jitter_saturation": args.augment_color_jitter_saturation,
            "color_jitter_contrast": args.augment_color_jitter_contrast,
            "brightness_jitter": args.augment_brightness_jitter,
            "brightness_jitter_delta": args.augment_brightness_jitter_delta,
        },
        "training": {
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "unfreeze_lr": args.unfreeze_lr,
            "weight_decay": args.weight_decay,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "unfreeze_last_block": args.unfreeze_last_block,
            "freeze_batchnorm": args.freeze_batchnorm,
            "amp": args.amp,
        },
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "splits": {
            "train_base": len(train_samples),
            "val_base": len(val_samples),
            "test_base": len(test_samples),
        },
        "variants": {
            "train": len(train_variants),
            "val": len(val_variants),
            "test": len(test_variants),
        },
    }

    with (run_root / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessing_payload, handle, indent=2)

    write_dataset_manifest(
        run_root / "dataset_manifest.csv",
        splits={
            "train": train_variants,
            "val": val_variants,
            "test": test_variants,
        },
        label_texts=label_texts,
    )

    summary: List[SummaryRow] = []
    for model_name in args.models:
        model_dir = run_root / model_name
        result = train_one_model(
            model_name,
            args,
            train_variants,
            val_variants,
            test_variants,
            model_dir,
            label_texts,
        )
        summary.append(result)

    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\\nSummary:")
    for row in summary:
        print(
            f"- {row['model']}: val_f1={row['val_f1']:.4f} "
            f"test_f1={row['test_f1']:.4f} "
            f"test_acc={row['test_accuracy']:.4f}"
        )
    print(f"Outputs saved to: {run_root}")


if __name__ == "__main__":
    main()
