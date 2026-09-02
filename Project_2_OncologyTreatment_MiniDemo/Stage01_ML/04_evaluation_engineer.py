"""
ROLE: EVALUATION ENGINEER
JOB (from problem statement): "Stress-test the model against unseen past
patient cohorts to flag overconfidence."

WHY THIS STEP EXISTS:
A model that looks good is not the same as a model that IS good. This
role checks the model on rows it has NEVER seen before (the test set),
and flags cases where the model is confidently WRONG - dangerous in a
real tumor board decision.
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = joblib.load("data/risk_model.pkl")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()  # squeeze -> turn single column into a Series

# STEP 1: Predict on the unseen test rows
y_pred = model.predict(X_test)

# STEP 2: Basic accuracy - how many did it get right?
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on unseen test rows: {acc:.0%}")

# STEP 3: Confusion matrix - WHERE did it get confused (which labels mixed up)?
print("\nConfusion matrix (rows=actual, columns=predicted):")
labels = sorted(y_test.unique())
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels),
                    index=labels, columns=labels))

# STEP 4: Full precision/recall report
print("\nDetailed report:")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 5: Flag "overconfidence" - predictions where the model was VERY sure
# but WRONG (this is what the problem statement specifically asks us to check)
probabilities = model.predict_proba(X_test)
max_confidence = probabilities.max(axis=1)
for i, (actual, pred, conf) in enumerate(zip(y_test, y_pred, max_confidence)):
    flag = "  <-- OVERCONFIDENT & WRONG!" if (pred != actual and conf > 0.8) else ""
    print(f"Row {i}: actual={actual}, predicted={pred}, confidence={conf:.0%}{flag}")
