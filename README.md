# ImageValidationPipeline

End-to-end image analysis and validation pipeline for a bachelor project. The
main work focuses on detecting blurry images, validating whether image content is
usable, and comparing CNN fine-tuning strategies for binary image
classification.

## Highlights

- Fine-tuned and compared CNN backbones: ResNet18, ResNet50, and SqueezeNet.
- Used transfer learning from ImageNet and a domain-relevant DTD ResNet50
  checkpoint.
- Implemented stratified train/validation/test splits with label-balance
  reporting.
- Evaluated models with accuracy, precision, recall, F1, confusion counts, and
  per-epoch curves.
- Ran multi-seed and hyperparameter experiments, including learning-rate sweeps,
  freeze/unfreeze schedules, and epoch comparisons.
- Tested preprocessing and augmentation strategies such as blur, JPEG quality,
  noise, rotations, flips, brightness jitter, and color jitter.
- Saved reproducible experiment artifacts including `metrics.json`,
  `summary.json`, `dataset_manifest.csv`, split reports, plots, and review
  grids.

Best reported result: ResNet50 with DTD initialization reached about
`test_f1=0.9492` and `test_acc=0.9348` in the recorded experiments.

## Repository Navigation

- `Image_Processing/Content_validation/` - main supervised training and
  evaluation code for content validation. Start here for CNN fine-tuning.
- `Image_Processing/Content_validation/README.md` - detailed commands for setup,
  data checks, training, learning-rate sweeps, augmentation, and outputs.
- `Image_Processing/Content_validation/pipeline_steps/` - modular loading,
  preprocessing, training, and reporting steps used by the clean pipeline entry
  point.
- `Image_Processing/Content_validation/runs/` - saved experiment outputs,
  metrics, manifests, and model checkpoints.
- `Image_Processing/Blurr_detection/` - blur-related ground truth, notes, and
  review artifacts.
- `src/` - blur-gate pipeline components, including full-image, ROI patch, SAM +
  Laplacian, and review-plot scripts.
- `reports/` - exported tables and report artifacts.
- `summary.md` and `runresults.md` - human-readable summaries of important
  experiment results.
- `Datasetdokumentation/` - dataset documentation and data-card export tooling.
- `checkpoints/` - external or pretrained checkpoint files used by experiments.

## Environment

This project uses Python with `uv`. The current virtual environment path is:

```bash
.venv
```

Create and activate it from the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision pillow matplotlib
```

Optional packages may be needed for broader model experiments or reporting, such
as `timm`, depending on the script being run.

## Common Workflow

Run commands from the repository root.

Check ground-truth completeness:

```bash
uv run python Image_Processing/Content_validation/check_ground_truth.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --label-column "usability considering nfp" \
  --filename-column filename \
  --match-mode filename \
  --decode-percent-newlines
```

Train one content-validation model:

```bash
uv run python Image_Processing/Content_validation/main.py \
  --csv-path ground_truth.csv \
  --image-root Images \
  --project-column project \
  --filename-column filename \
  --label-column "usability considering nfp" \
  --positive-labels "usable" \
  --negative-labels "not usable" \
  --models resnet50 \
  --weights imagenet \
  --device auto \
  --decode-percent-newlines \
  --save-val-grid
```

For more complete examples, see
`Image_Processing/Content_validation/README.md`.

## What To Know Before Running

- The raw image folders and some ground-truth CSV files may not be committed to
  the repository. Training commands expect paths such as `Images/` and
  `ground_truth.csv` to exist locally.
- Some experiments depend on pretrained checkpoints in `checkpoints/`, including
  the DTD ResNet50 checkpoint.
- Runs can be long. The content-validation README recommends running one model
  per command to avoid long multi-model jobs.
- Validation metrics are used for checkpoint selection; test metrics are
  computed from the selected checkpoint.
- Existing run folders are experiment records. Avoid overwriting them unless the
  goal is to intentionally replace an experiment.

## Main Technologies

- Python
- PyTorch and torchvision
- uv
- Pillow
- Matplotlib
- CNN transfer learning and fine-tuning
