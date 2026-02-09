# Blur Thresholds (Size-Aware Label Switching)

This document summarizes the **deterministic blur label-flip rules** used in the
content-validation pipeline.

## Rule
A blur-augmented sample **keeps** its original label until the blur radius is
**greater than** the size-category threshold. If the blur radius is **greater
than** the threshold, the label flips to **"not usable"**.

## Size categories and thresholds

Pixel size is based on **width × height**, reported in megapixels (MP).

| Size category | Pixel size (MP) | Flip threshold (Gaussian blur radius) | Label behavior |
| --- | --- | --- | --- |
| Small | <= 2 MP | > 2 | keep if <= 2, flip if > 2 |
| Medium | <= 10 MP | > 10 | keep if <= 10, flip if > 10 |
| Very large | > 10 MP | > 20 | keep if <= 20, flip if > 20 |

## Deterministic vs random blur
- **Deterministic blur variants** are generated when
  `--augment-blur-keep-range` / `--augment-blur-flip-range` are supplied.
- **Random blur** is enabled with `--augment-random-blur` and uses the same
  thresholds to decide if the label flips when `--augment-blur-size-aware` is on.
- Random blur avoids radii within the **reject radius** (default 4) around the
  flip threshold to keep label flips unambiguous.
- If `--augment-blur-only-positive` is enabled, only **usable** images are
  eligible for label flipping.

## Related flags (for reference)
- `--augment-blur-size-aware`
- `--blur-size-small-max-mp`
- `--blur-size-medium-max-mp`
- `--blur-switch-small`
- `--blur-switch-medium`
- `--blur-switch-large`
- `--augment-random-blur-reject-radius`
