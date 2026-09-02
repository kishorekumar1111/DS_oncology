"""
ROLE: INTEGRATION ENGINEER (GenAI stage)
JOB: "Integrate synthetic scenario output back into the testing dashboard
for continuous evaluation."

WHY THIS STEP EXISTS:
The whole point of generating synthetic rare-mutation patients is to feed
them BACK into the real pipeline (Stage 1's ML model) as a stress test.
This closes the loop between GenAI and ML.
"""

import pandas as pd
import os

synthetic = pd.read_csv("data/synthetic_scenarios.csv")

# Try to reuse the REAL trained model from Stage 1 (shows how stages connect)
stage1_model_path = "../Stage01_ML/data/risk_model.pkl"

if os.path.exists(stage1_model_path):
    import joblib
    model = joblib.load(stage1_model_path)
    print("Loaded the real Stage 1 ML risk model to stress-test with synthetic patients.\n")

    for i, row in synthetic.iterrows():
        # Stage 1 model expects: tumor_mutation_burden, ctdna_level_ng_ml, creatinine_mg_dl, alt_u_l
        input_row = pd.DataFrame([[row["tumor_mutation_burden"], row["ctdna_level_ng_ml"],
                                    row["creatinine_mg_dl"], 45]],  # assume mid-range ALT for this test
                                  columns=["tumor_mutation_burden", "ctdna_level_ng_ml",
                                           "creatinine_mg_dl", "alt_u_l"])
        prediction = model.predict(input_row)[0]
        match = "MATCHES expected label" if prediction == row["risk_label"] else "MISMATCH - investigate!"
        print(f"Synthetic scenario {i+1}: expected={row['risk_label']}, "
              f"model predicted={prediction} -> {match}")
else:
    print("(Run Stage01_ML first to enable full cross-stage stress testing.)")
    print("Showing synthetic scenarios only:\n")
    print(synthetic)

print("\nThis is how Generative AI stress-tests the earlier ML stage before real deployment.")
