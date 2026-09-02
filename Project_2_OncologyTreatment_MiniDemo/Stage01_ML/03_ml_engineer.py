"""
ROLE: ML ENGINEER
JOB (from problem statement): "Train and tune the core classifier scoring
patient toxicity risk in real time."

WHY THIS STEP EXISTS:
This is the step most people call "doing ML" - but notice it only works
BECAUSE two roles already did their job (clean data + verified data).

ALGORITHM: Decision Tree Classifier - chosen because it is easy to explain
to an oncologist (it literally asks yes/no questions about biomarkers, like
a human tumor board would) and matches the study material's "classification
fundamentals, decision trees" requirement.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/eda_checked_dataset.csv")

# STEP 1: Choose the FEATURES (inputs) and the TARGET (what we want to predict)
features = ["tumor_mutation_burden", "ctdna_level_ng_ml", "creatinine_mg_dl", "alt_u_l"]
X = df[features]
y = df["risk_label"]

# STEP 2: Split data - some rows to TRAIN on, some rows to TEST on later
# (small dataset, so we keep the split simple: 80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 3: Train (fit) the Decision Tree on the training rows
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth=3 keeps it simple/explainable
model.fit(X_train, y_train)

# STEP 4: Quick sanity check - what does the tree think matters most?
print("Feature importance (which biomarker matters most for toxicity risk):")
for feat, importance in zip(features, model.feature_importances_):
    print(f"  {feat}: {importance:.2f}")

# STEP 5: Save the trained model AND the test set for the Evaluation Engineer
joblib.dump(model, "data/risk_model.pkl")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nModel trained and saved to data/risk_model.pkl")
print("Test rows saved for the Evaluation Engineer to check next.")
