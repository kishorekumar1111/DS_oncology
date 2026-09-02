"""
ROLE: EVALUATION ENGINEER (GenAI stage)
JOB: "Audit generated synthetic patient profiles to verify they realistically
stress agent decision logic."

WHY THIS STEP EXISTS:
Generated data is only useful if it's REALISTIC. A scenario with negative
tumor mutation burden or an impossible creatinine level would be useless
(or misleading) for testing. This role sanity-checks the output.
"""

import pandas as pd

df = pd.read_pickle("data/historical_df.pkl")
synthetic = pd.read_csv("data/synthetic_scenarios.csv")

print("Sanity checks on generated patient scenarios:\n")
for i, row in synthetic.iterrows():
    checks = []
    checks.append("tumor mutation burden in plausible range" if 0 <= row["tumor_mutation_burden"] <= 40 else "UNREALISTIC tumor mutation burden!")
    checks.append("ctDNA level in plausible range" if 0 <= row["ctdna_level_ng_ml"] <= 30 else "UNREALISTIC ctDNA level!")
    checks.append("creatinine in plausible range" if 0 <= row["creatinine_mg_dl"] <= 6 else "UNREALISTIC creatinine!")

    print(f"Scenario {i+1}: {dict(row)}")
    for c in checks:
        print(f"   - {c}")
    print()

print("If all checks pass, these scenarios are ready to stress-test the ML/DL/NLP pipeline.")
