"""
ROLE: DATA ENGINEER (GenAI stage)
JOB: "Assemble reference baseline datasets from real historical oncology
distributions."

WHY THIS STEP EXISTS:
Before generating FAKE (synthetic) rare-mutation patients, we must
understand what REAL patients look like statistically - otherwise the fake
ones won't be realistic.
"""

import pandas as pd

df = pd.read_csv("data/historical_genomic_stats.csv")

print("Historical genomic data loaded:")
print(df)

# STEP: Compute the statistical baseline (mean + spread) per risk level
# This baseline is what the GenAI Engineer will sample new patient scenarios from.
baseline = df.groupby("risk_label")[["tumor_mutation_burden", "ctdna_level_ng_ml", "creatinine_mg_dl"]].agg(["mean", "std"])
print("\nStatistical baseline per risk level (mean, std):")
print(baseline)

baseline.to_pickle("data/baseline_stats.pkl")
df.to_pickle("data/historical_df.pkl")
print("\nSaved baseline to data/baseline_stats.pkl")
