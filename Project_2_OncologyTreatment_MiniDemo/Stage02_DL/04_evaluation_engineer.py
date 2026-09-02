"""
ROLE: EVALUATION ENGINEER (Deep Learning stage)
JOB: "Refine confusion matrices until the model stops mistaking benign
inflammation for malignant progression."

WHY THIS STEP EXISTS:
A vision model that confuses "benign inflammation" with "malignant tumor
margin" could send an oncologist down the wrong treatment path entirely.
We check exactly WHERE it gets confused.
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

vision_model = joblib.load("data/vision_model.pkl")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

y_pred = vision_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Vision model accuracy on unseen test tiles: {acc:.0%}")

labels = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(f"{'':12}" + "".join(f"{l:>12}" for l in labels))
for i, row_label in enumerate(labels):
    print(f"{row_label:12}" + "".join(f"{v:>12}" for v in cm[i]))

# Flag the exact failure mode the problem statement warns about
false_negatives = ((y_test == "Malignant") & (y_pred == "Benign")).sum()
print(f"\nDangerous misses (real malignant tile called 'Benign'): {false_negatives}")
print("This is the number the team must drive to ZERO before real clinical deployment.")
