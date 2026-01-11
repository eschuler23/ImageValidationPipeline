# Project goal

This project focuses on analyzing image data and drawing conclusions from that analysis.

The work in `Image_Processing/` is organized around three goals:
1. Detect and sort blurry images using a ground truth reference (for example, `groundtruth.csv`).
2. Evaluate lighting quality (good vs. bad) where we do not yet have a defined approach.
3. Validate image content using more complex methods (for example, training models); details will be captured in the specific `.md` files later.

## Data science workflow expectations
- Use `uv` (Astral) with Python for data science tasks (data loading, math, image inspection).
- Always use a VINV and keep it consistent across analyses.
- Once a VINV is created in this folder, record its path here to avoid confusion:
  - VINV path: TBD
- Prefer using bash commands to drive `uv` so analysis is reproducible and iterative.
