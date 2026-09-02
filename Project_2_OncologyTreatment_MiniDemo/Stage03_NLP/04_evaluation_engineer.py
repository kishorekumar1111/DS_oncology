"""
ROLE: EVALUATION ENGINEER (NLP stage)
JOB: "Benchmark precision/recall on held-out clinical trial logs; analyze
edge-case failures."

WHY THIS STEP EXISTS:
Missing a real "Urgent" clinical note (calling it Routine) could delay a
critical adverse-event response. We specifically check for this failure type.
"""

import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/clean_logs.csv")
vectorizer = joblib.load("data/vectorizer.pkl")
clf = joblib.load("data/urgency_classifier.pkl")

X = vectorizer.transform(df["clean_message"])
y_true = df["urgency_label"]
y_pred = clf.predict(X)

print("Classification report:")
print(classification_report(y_true, y_pred, zero_division=0))

# STEP: Flag the dangerous failure type specifically
missed_urgent = df[(y_true == "Urgent") & (y_pred == "Routine")]
print(f"\nDangerous misses (real URGENT note called Routine): {len(missed_urgent)}")
if len(missed_urgent):
    print(missed_urgent[["message"]])
else:
    print("None in this mini test - good sign, but the real dataset needs much more testing.")
