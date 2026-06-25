#!/usr/bin/env python3
"""Generate a clean schematic diagram of the crop augmentation strategy for the paper.

Creates a diagram showing:
- 512x512 original image
- 384x384 central region (offset 64 from edges)
- 3x3 grid of 256x256 crop positions (stride 64)
- 10th option: full-image resize to 256x256

Output: results/figures/crop_augmentation.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

# Configuration matching the dataset
ORIGINAL_SIZE = 512
CROP_SIZE = 256
CENTER_REGION_SIZE = 384
CENTER_OFFSET = (ORIGINAL_SIZE - CENTER_REGION_SIZE) // 2  # 64

# Crop positions (y, x) in 512x512 space
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

# Output path
OUTPUT_DIR = Path("results/figures")
OUTPUT_FILE = OUTPUT_DIR / "crop_augmentation.png"


def draw_crop_panel(ax: plt.Axes, crop_indices: list[int], title: str) -> None:
    """Draw a panel showing specific crop positions.

    Args:
        ax: matplotlib axis
        crop_indices: list of crop indices to draw
        title: panel title

    """
    ax.set_xlim(-30, ORIGINAL_SIZE + 20)
    ax.set_ylim(ORIGINAL_SIZE + 20, -30)  # Invert y-axis (image coordinates)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw original 512x512 image outline
    original_rect = patches.Rectangle(
        (0, 0),
        ORIGINAL_SIZE,
        ORIGINAL_SIZE,
        linewidth=2,
        edgecolor="black",
        facecolor="white",
        zorder=1,
    )
    ax.add_patch(original_rect)

    # Draw 384x384 central region with light gray fill
    center_rect = patches.Rectangle(
        (CENTER_OFFSET, CENTER_OFFSET),
        CENTER_REGION_SIZE,
        CENTER_REGION_SIZE,
        linewidth=1.5,
        edgecolor="gray",
        facecolor="#e8e8e8",
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(center_rect)

    # Draw selected 256x256 crop rectangles
    colors = plt.cm.tab10.colors
    for idx in crop_indices:
        y, x = CROP_POSITIONS[idx]
        crop_rect = patches.Rectangle(
            (x, y),
            CROP_SIZE,
            CROP_SIZE,
            linewidth=2,
            edgecolor=colors[idx % len(colors)],
            facecolor=(*colors[idx % len(colors)][:3], 0.15),  # Light fill
            zorder=3,
        )
        ax.add_patch(crop_rect)

        # Add crop number label
        ax.text(
            x + CROP_SIZE / 2,
            y + CROP_SIZE / 2,
            str(idx),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=colors[idx % len(colors)],
            zorder=4,
        )

    # Add dimension labels
    ax.annotate(
        "",
        xy=(ORIGINAL_SIZE, -10),
        xytext=(0, -10),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.5},
    )
    ax.text(ORIGINAL_SIZE / 2, -18, "512", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate(
        "",
        xy=(-10, ORIGINAL_SIZE),
        xytext=(-10, 0),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.5},
    )
    ax.text(-18, ORIGINAL_SIZE / 2, "512", ha="right", va="center", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def create_crop_diagram() -> None:
    """Create the crop augmentation schematic diagram."""
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    ax1, ax2, ax3 = axes

    # Split crops into two groups to avoid overlap confusion:
    # Group 1: Center + corners (0, 5, 6, 7, 8) - these form an X pattern
    # Group 2: Edges (1, 2, 3, 4) - these form a + pattern
    corners_center = [0, 5, 6, 7, 8]  # Center, TL, TR, BL, BR
    edges = [1, 2, 3, 4]  # Top, Bottom, Left, Right

    # Panel (a): Corners + center
    draw_crop_panel(ax1, corners_center, "(a) Center + corners (0,5,6,7,8)")

    # Panel (b): Edges
    draw_crop_panel(ax2, edges, "(b) Edge crops (1,2,3,4)")

    # === Panel (c): Full-image resize option ===
    # Use a smaller scale for the 512 box to fit both in view
    scale = 0.6  # Scale factor for visual representation
    orig_size_scaled = ORIGINAL_SIZE * scale  # 307.2
    resized_size_scaled = 256 * scale  # 153.6

    gap = 60  # Gap between the two boxes
    total_height = orig_size_scaled + gap + resized_size_scaled

    ax3.set_xlim(-40, orig_size_scaled + 40)
    ax3.set_ylim(total_height + 40, -40)
    ax3.set_aspect("equal")
    ax3.axis("off")

    # Draw original 512x512 outline (scaled)
    orig_rect = patches.Rectangle(
        (0, 0),
        orig_size_scaled,
        orig_size_scaled,
        linewidth=2,
        edgecolor="black",
        facecolor="#f5f5f5",
        zorder=1,
    )
    ax3.add_patch(orig_rect)

    # Label for original
    ax3.text(
        orig_size_scaled / 2,
        -8,
        "512 × 512",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # Arrow pointing down
    arrow_start_y = orig_size_scaled + 10
    arrow_end_y = orig_size_scaled + gap - 10
    ax3.annotate(
        "",
        xy=(orig_size_scaled / 2, arrow_end_y),
        xytext=(orig_size_scaled / 2, arrow_start_y),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )
    ax3.text(
        orig_size_scaled / 2 + 8,
        (arrow_start_y + arrow_end_y) / 2,
        "resize",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Draw resized 256x256 representation (scaled, centered)
    resize_offset_x = (orig_size_scaled - resized_size_scaled) / 2
    resize_offset_y = orig_size_scaled + gap
    resized_rect = patches.Rectangle(
        (resize_offset_x, resize_offset_y),
        resized_size_scaled,
        resized_size_scaled,
        linewidth=2,
        edgecolor="black",
        facecolor="white",
        zorder=1,
    )
    ax3.add_patch(resized_rect)

    # Label for resized
    ax3.text(
        orig_size_scaled / 2,
        resize_offset_y + resized_size_scaled + 8,
        "256 × 256",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    ax3.set_title("(c) Global resize", fontsize=11, fontweight="bold", pad=8)

    # Add shared info at bottom
    fig.text(
        0.5,
        0.02,
        "Crop size: 256 × 256  |  Central region: 384 × 384  |  Stride: 64",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"Diagram saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_crop_diagram()
