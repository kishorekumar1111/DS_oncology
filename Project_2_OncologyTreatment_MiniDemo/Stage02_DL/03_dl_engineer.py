"""
ROLE: DL ENGINEER
JOB: "Build and train the CNN vision network and LSTM/Transformer sequence
model."

WHY THIS STEP EXISTS:
Two separate deep learning jobs happen in this stage:
 1. VISION: look at a pathology tile, say Malignant or Benign (CNN's job)
 2. SEQUENCE: look at past ctDNA readings, project the trend forward (LSTM's job)

SIMPLIFIED FOR CLASSROOM: We use scikit-learn's MLPClassifier (a small
neural network) as a stand-in for a real CNN, and simple linear trend
fitting as a stand-in for a real LSTM/Transformer. Same INPUT->OUTPUT
shape as the real thing, tiny enough to train instantly.
"""

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ---------- PART A: Vision model (CNN stand-in) ----------
images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42
)

vision_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=42)
vision_model.fit(X_train, y_train)
joblib.dump(vision_model, "data/vision_model.pkl")
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
print(f"Vision model trained. Training accuracy: {vision_model.score(X_train, y_train):.0%}")

# ---------- PART B: Sequence model (LSTM/Transformer stand-in) ----------
# Toy ctDNA readings over the last 6 monthly draws for one patient (rising trend)
ctdna_readings = np.array([1.2, 1.8, 2.6, 3.9, 5.1, 6.2])
months = np.arange(len(ctdna_readings)).reshape(-1, 1)

# A simple straight-line trend fit stands in for a real LSTM/Transformer,
# which would learn much more complex, non-straight-line patterns.
slope, intercept = np.polyfit(months.flatten(), ctdna_readings, 1)
next_3_months = [intercept + slope * (len(ctdna_readings) + m) for m in range(1, 4)]

print(f"\nctDNA trend so far (ng/mL): {ctdna_readings}")
print(f"Projected ctDNA level, next 3 months: {[round(v, 2) for v in next_3_months]}")

joblib.dump({"slope": slope, "intercept": intercept}, "data/sequence_model.pkl")
