"""
ROLE: EDA ENGINEER (Deep Learning stage)
JOB: "Inspect model saliency maps to verify it focuses on cellular atypia,
not slide artifacts."

WHY THIS STEP EXISTS:
Before trusting a vision model, we must check WHERE in the tile it's
looking. A real project uses "saliency maps" (heatmaps of important
pixels). Here we do the simplified version: compare average brightness
of the TOP half vs BOTTOM half of each tile, split by label - to confirm
our tiles actually differ where we expect (bottom = nuclei cluster).
"""

import numpy as np

images = np.load("data/images.npy")   # shape: (N, 64) flattened 8x8
labels = np.load("data/labels.npy")

# STEP 1: Reshape back to 8x8 so we can look at top vs bottom half
images_2d = images.reshape(len(images), 8, 8)

top_half_brightness = images_2d[:, :4, :].mean(axis=(1, 2))
bottom_half_brightness = images_2d[:, 4:, :].mean(axis=(1, 2))

print("Average BOTTOM-HALF brightness (where the nuclei cluster would appear):")
for label in np.unique(labels):
    mask = labels == label
    print(f"  {label}: {bottom_half_brightness[mask].mean():.2f}")

print("\nAverage TOP-HALF brightness (should look similar for both classes):")
for label in np.unique(labels):
    mask = labels == label
    print(f"  {label}: {top_half_brightness[mask].mean():.2f}")

# STEP 2: Sanity check - the model SHOULD be learning from the bottom half
# (real cellular atypia), not some unrelated staining artifact in the top half.
gap = bottom_half_brightness[labels == "Malignant"].mean() - bottom_half_brightness[labels == "Benign"].mean()
print(f"\nBottom-half brightness gap between classes: {gap:.2f}")
print("A clear positive gap here means: yes, the difference is really about cell density, not artifact noise.")
