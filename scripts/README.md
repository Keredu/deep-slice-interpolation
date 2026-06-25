# Scripts

Utility scripts for preprocessing, export, and visualization.

## Preprocessing

### preprocess_raw_kaggle_rsna2019_IH.py

Converts raw DICOM files from the RSNA Intracranial Hemorrhage Detection dataset to windowed PNG images.

```bash
uv run scripts/preprocess_raw_kaggle_rsna2019_IH.py
```

**Input**: `$DATASETS_DIR/raw/rsna-intracranial-hemorrhage-detection/`
**Output**: `$DATASETS_DIR/pre/rsna-intracranial-hemorrhage-detection/1x512x512_-20_107/`

**Windowing Parameters**:
- Window center: 44 HU
- Window width: 128 HU
- Effective range: -20 to 107 HU (128 discrete values, optimized for brain tissue)

**Creates**:
- PNG images organized by patient ID
- Metadata CSV (`df.csv`) with IDs, ordering, and train/valid/test splits

## Data Inspection

### get_test_patients.py

Extracts test-split patients' files for manual inspection.

```bash
uv run scripts/get_test_patients.py
```

**Output**: `output/raw-test/{dataset_name}/{patient_id}/{img,dcm}/`


## Visualization

### generate_missing_viz.py

Batch generates missing `target_is_real` visualizations across experiments. Useful when training was done with `generate_test_viz_real=False`.

```bash
# Process all experiments
uv run scripts/generate_missing_viz.py

# Process specific experiment
uv run scripts/generate_missing_viz.py exp_name

# Dry run (show what would be done)
uv run scripts/generate_missing_viz.py --dry-run

# Custom experiments directory
uv run scripts/generate_missing_viz.py --experiments-dir experiments/train_nn1_cropped
```

**Scans**: All epoch directories with `weights.pth` but no `viz/target_is_real/`
**Output**: `{epoch_dir}/viz/target_is_real/`

### plot_loss_curves.py

Plots train and validation loss curves from `epochs.csv`.

```bash
# Plot for all experiments
uv run scripts/plot_loss_curves.py

# Plot for specific experiment
uv run scripts/plot_loss_curves.py exp_name

# Custom experiments directory
uv run scripts/plot_loss_curves.py --experiments-dir experiments/train_nn1_cropped
```

**Input**: `{experiment_dir}/epochs.csv`
**Output**: `{experiment_dir}/loss_curves.png`

Features:
- Marks best epochs with green dots
- Uses log scale when loss range > 10x

### visualize_crops.py

Generates visualizations of the 9-crop augmentation strategy.

```bash
uv run scripts/visualize_crops.py
```

**Output**: `output/crop_visualizations/`

Shows:
- Crop position diagram on full 512x512 image
- Extracted 256x256 crops from center 384x384 region

## Export

### export_test_interpolated_series.py

Exports per-patient test interpolation results as:
- PNG images
- DICOM Secondary Capture series (preserves original geometry/metadata)

```bash
uv run scripts/export_test_interpolated_series.py
```

**Requires**: Trained model checkpoint
**Output**: `output/test_interpolated_export/{patient_id}/`

## Maintenance

### cleanup_old_epochs.py

Removes low-score epoch directories to save disk space.

```bash
uv run scripts/cleanup_old_epochs.py --experiment_dir experiments/... --keep_best 5
```

