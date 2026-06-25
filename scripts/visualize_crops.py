#!/usr/bin/env python3
"""Visualize crop positions on sample images from the dataset.

Simplified version that:
- Loads images directly from the dataset directory (no dataset class)
- Uses only standard libraries (PIL, numpy, matplotlib)
- Outputs to output/crop_visualizations/

Strategy:
- Focus on center 384×384 region of 512×512 image
- Create 9 crops of 256×256 with stride 64

Shows:
1. Original 512×512 image with red rectangles and numbers showing crop positions
2. Collage with downsampled original + all 9 crops (2×5 grid)
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf
from dotenv import load_dotenv
from matplotlib import patches
from PIL import Image

load_dotenv()

# Crop configuration (matching the dataset class)
CROP_SIZE = 256
CENTER_REGION_SIZE = 384
CENTER_OFFSET = (512 - CENTER_REGION_SIZE) // 2  # 64 pixels

# Crop positions in original 512×512 space
# Focus on center 384×384 region with stride 64
CROP_POSITIONS = [
    (128, 128),  # 0: Center
    (128, 64),  # 1: Top
    (128, 192),  # 2: Bottom
    (64, 128),  # 3: Left
    (192, 128),  # 4: Right
    (64, 64),  # 5: Top-left
    (192, 64),  # 6: Top-right
    (64, 192),  # 7: Bottom-left
    (192, 192),  # 8: Bottom-right
]

# Configuration
DATASET_NAME = "1x512x512_-20_107"
NUM_SAMPLES = 5
OUTPUT_DIR = Path("./output/crop_visualizations")


def get_image_dataset_dir() -> Path:
    """Return the configured preprocessed image directory."""
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        raise RuntimeError("DATASETS_DIR is not set. Define it in .env or the shell.")
    return Path(datasets_dir) / "pre" / "rsna-intracranial-hemorrhage-detection" / DATASET_NAME


def create_crop_collage(img_arr: np.ndarray, pil_img: Image.Image, save_path: Path) -> None:
    """Create detailed collage showing original + each crop position.

    Layout: 10 rows x 2 columns
    - Row 0: Original image
    - Rows 1-9: [Original with red box] + [Extracted crop]

    Args:
        img_arr: numpy array for cropping
        pil_img: PIL Image (not used in new version)
        save_path: where to save
    """
    _fig, axes = plt.subplots(10, 2, figsize=(12, 50))

    # Row 0: Original image (use first column, hide second)
    axes[0, 0].imshow(img_arr, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original 512×512 Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")  # Hide right column in first row

    # Rows 1-9: Loop through each crop position
    for idx, (y, x) in enumerate(CROP_POSITIONS):
        row = idx + 1  # Row index (1-9)

        # Left subplot: Original with red box at this position
        ax_left = axes[row, 0]
        ax_left.imshow(img_arr, cmap="gray", vmin=0, vmax=1)

        # Draw red rectangle at this crop position
        rect = patches.Rectangle(
            (x, y),
            CROP_SIZE,
            CROP_SIZE,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax_left.add_patch(rect)

        # Add number label
        ax_left.text(
            x + 10,
            y + 20,
            str(idx),
            color="red",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        ax_left.set_title(f"Crop {idx} Position", fontsize=10)
        ax_left.axis("off")

        # Right subplot: Extracted crop using tf.crop
        ax_right = axes[row, 1]
        crop_img = tf.crop(pil_img, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
        crop_arr = np.array(crop_img, dtype=np.float32) / 255.0
        ax_right.imshow(crop_arr, cmap="gray", vmin=0, vmax=1)
        ax_right.set_title(f"Crop {idx}: ({x},{y})", fontsize=10)
        ax_right.axis("off")

    # Save figure
    plt.suptitle("Crop Positions and Extracted Regions (Center 384×384)", fontsize=16, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.985])  # Leave space at top for title
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Generate visualizations for sample images."""
    img_dataset_dir = get_image_dataset_dir()
    if not img_dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {img_dataset_dir}")

    # Remove and recreate output directory to ensure clean state
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all PNG files from dataset directory
    all_images = sorted(img_dataset_dir.glob("*.png"))
    if len(all_images) == 0:
        raise RuntimeError(f"No PNG files found in {img_dataset_dir}")

    # Select random samples
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_images), size=min(NUM_SAMPLES, len(all_images)), replace=False)

    for i, idx in enumerate(sample_indices):
        img_path = all_images[idx]

        # Load image
        with Image.open(img_path) as pil_img:
            img_arr = np.array(pil_img, dtype=np.float32) / 255.0

            # Create collage visualization
            collage_path = OUTPUT_DIR / f"sample_{i:02d}_collage.png"
            create_crop_collage(img_arr, pil_img, collage_path)


if __name__ == "__main__":
    main()
