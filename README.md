# Deep Slice Interpolation

Python code and lightweight experiment records for CT slice interpolation from
neighboring axial slices. The project trains U-Net models with an
EfficientNetV2 encoder to predict the middle slice from adjacent CT slices.

This public repository is intended for running new experiments and inspecting
the training records used in the study. It does not include private writing
materials, raw medical images, generated DICOM exports, qualitative panels, or
model checkpoints.

## Included Artifacts

- `phd/`: Python package for datasets, losses, metrics, models, and training.
- `scripts/`: preprocessing, evaluation, plotting, export, and analysis tools.
- `tests/`: unit tests for the reusable Python code.
- `experiments/`: experiment registry, configs, epoch CSVs, and selected
  tracked loss curves/logs.
- `datasets/pre/rsna-intracranial-hemorrhage-detection/df.csv`: split metadata
  and slice ordering derived from the RSNA Intracranial Hemorrhage Detection
  dataset.

Model weights (`*.pth`), raw DICOMs, preprocessed PNG images, generated figures,
and aggregate paper table CSVs are intentionally excluded.

## Installation

The project uses Python 3.13 and `uv`.

```bash
git clone https://github.com/Keredu/deep-slice-interpolation.git
cd deep-slice-interpolation
uv sync
cp example.env .env
```

Edit `.env` so `DATASETS_DIR` points to a local dataset root:

```bash
DATASETS_DIR="/path/to/datasets"
```

## Dataset Setup

Download the RSNA Intracranial Hemorrhage Detection dataset from Kaggle and
place the raw files under:

```text
$DATASETS_DIR/raw/rsna-intracranial-hemorrhage-detection/
```

Then create the windowed PNG dataset:

```bash
uv run scripts/preprocess_raw_kaggle_rsna2019_IH.py
```

The training code expects:

```text
$DATASETS_DIR/pre/rsna-intracranial-hemorrhage-detection/1x512x512_-20_107/
```

The tracked `df.csv` preserves the train/validation/test split metadata used by
the experiments.

## Running Experiments

Register the configured experiments, then run one or all queued experiments:

```bash
uv run register_experiments.py --show
uv run register_experiments.py
uv run train.py --show-queue
uv run train.py
uv run train.py --run-all
```

Training outputs are written under `experiments/train_nn1_cropped/{experiment}`.
Each experiment stores `config.json`, `epochs.csv`, checkpoints, and optional
visualizations. Checkpoints and generated visualizations are ignored by git.

## Reproducing Study Runs

The archived experiment folders contain the lightweight records needed to audit
training behavior:

- `config.json`: model, optimizer, scheduler, loss, augmentation, and runtime
  settings.
- `epochs.csv`: per-epoch losses, metrics, timings, learning rates, and best
  epoch flags.
- selected `training.log` and `loss_curves.png` files for representative runs.

The archived configs use `${DATASETS_DIR}` for the dataset root. Set
`DATASETS_DIR` in `.env` before running scripts that reload those configs.

Scripts that evaluate checkpoints or export DICOM series require trained
weights. Those weights are not part of this repository.

## Useful Commands

```bash
# Run tests
uv run pytest --no-cov

# Run the default test suite with coverage
uv run pytest

# Lint
uv run ruff check .

# Generate loss curves from tracked epoch CSVs
uv run scripts/plot_loss_curves.py

# Serve the local experiment dashboard
uv run report.py
```

## Repository Layout

```text
phd/
  datasets/       Dataset implementations for slice triplets
  losses/         L1, SSIM, MS-SSIM, and combined losses
  metrics/        Image quality and error metrics
  models/         U-Net interpolation model setup
  training/       Config, registry, scheduler, checkpoint, and trainer code
scripts/          Reproducibility and analysis utilities
tests/            Unit tests
experiments/      Lightweight experiment records
datasets/pre/     Public split metadata only
```

## License and Citation

License and citation metadata will be added with the publication-ready release.
