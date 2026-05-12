# ImageValidationPipeline
Built an end-to-end supervised image validation pipeline.
Fine-tuned and compared CNN backbones: ResNet18, ResNet50, SqueezeNet.
Used transfer learning from both ImageNet and a domain-relevant DTD ResNet50 checkpoint.
Implemented stratified train/validation/test splits and tracked label balance.
Evaluated with accuracy, precision, recall, F1, confusion counts, and per-epoch curves.
Ran multi-seed and hyperparameter experiments, including learning-rate sweeps, freeze/unfreeze schedules, and epoch comparisons.
Added image preprocessing and augmentation experiments: blur, JPEG quality, noise, rotations, flips, brightness/color jitter.
Created reproducible experiment artifacts: metrics.json, summary.json, dataset_manifest.csv, split reports, plots, and review grids.
Best reported result appears to be around test_f1=0.9492, test_acc=0.9348 for ResNet50 with DTD initialization.
