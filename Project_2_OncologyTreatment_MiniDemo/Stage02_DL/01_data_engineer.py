"""
ROLE: DATA ENGINEER (Deep Learning stage)
JOB: "Label pathology tissue tiles; perform data augmentation for stain
variation and artifact conditions."

WHY THIS STEP EXISTS:
Real whole-slide histopathology images aren't available for this classroom
demo, so we GENERATE tiny 8x8-pixel synthetic "tissue tiles" that behave the
same way real ones would: malignant tiles have a denser, brighter cluster of
"cell nuclei" in the lower half (mimicking irregular tumor margin cell
density), benign tiles look more uniform. This keeps the workflow identical
to the real project, just at toy scale.
"""

import numpy as np

np.random.seed(42)


def make_tile(malignant: bool):
    """Creates one tiny 8x8 grayscale 'pathology tile'. malignant=True -> denser lower-half nuclei cluster."""
    img = np.random.uniform(0.1, 0.3, size=(8, 8))  # background tissue stain
    if malignant:
        img[4:, :] += np.random.uniform(0.4, 0.6, size=(4, 8))  # dense irregular nuclei cluster
    return np.clip(img, 0, 1)


# STEP 1: Generate a small labeled set of tiles (8 malignant, 8 benign)
tiles, labels = [], []
for _ in range(8):
    tiles.append(make_tile(malignant=True))
    labels.append("Malignant")
for _ in range(8):
    tiles.append(make_tile(malignant=False))
    labels.append("Benign")

tiles = np.array(tiles)
labels = np.array(labels)

# STEP 2: Simple data AUGMENTATION - simulate a stain-variation artifact by
# darkening a copy of every tile (this is what the problem statement asks
# for: "augmentation for stain variation and artifact conditions")
faint_stain_tiles = tiles * 0.5  # under-stained versions
tiles = np.concatenate([tiles, faint_stain_tiles])
labels = np.concatenate([labels, labels])

# STEP 3: Save as flattened rows (64 pixel values) + label, ready for training
flat = tiles.reshape(len(tiles), -1)  # 8x8 -> 64 values per tile
np.save("data/images.npy", flat)
np.save("data/labels.npy", labels)

print(f"Generated and labeled {len(tiles)} mini pathology tiles (with stain-variation augmentation).")
print(f"Class counts: Malignant={sum(labels=='Malignant')}, Benign={sum(labels=='Benign')}")
