"""
ROLE: EDA ENGINEER (Exploratory Data Analysis)
JOB (from problem statement): "Plot leading clinical indicators; hunt down
and eliminate historical false biomarker correlations."

WHY THIS STEP EXISTS:
Before training a model, we must LOOK at the data and check whether it
actually makes sense. If a "High Risk" label was recorded for a patient
whose biomarkers don't actually support it (a false correlation), and we
train on it anyway, the model learns the WRONG pattern - dangerous when
that pattern later decides real drug dosing.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets this run without a screen (saves plot to a file)
import matplotlib.pyplot as plt

df = pd.read_csv("data/master_dataset.csv")

# STEP 1: Basic pattern check - does higher tumor mutation burden generally
# mean higher toxicity risk?
print("Average tumor mutation burden per risk level:")
print(df.groupby("risk_label")["tumor_mutation_burden"].mean())

# STEP 2: Visualize tumor mutation burden vs ctDNA level, colored by risk label
# (This is exactly what the "Biomarker Leaderboard" will later be based on)
colors = {"Low": "green", "Moderate": "orange", "High": "red"}
plt.figure(figsize=(6, 4))
for label, group in df.groupby("risk_label"):
    plt.scatter(group["tumor_mutation_burden"], group["ctdna_level_ng_ml"],
                label=label, color=colors[label])
plt.xlabel("Tumor Mutation Burden (mut/Mb)")
plt.ylabel("ctDNA Level (ng/mL)")
plt.title("TMB vs ctDNA, colored by Toxicity Risk Level")
plt.legend()
plt.savefig("data/eda_pattern_plot.png")
print("\nPlot saved to data/eda_pattern_plot.png")

# STEP 3: Hunt down a "false biomarker correlation" - a row that doesn't fit
# the pattern. Example rule: labeled High risk but very low TMB/ctDNA is
# suspicious (could be a documentation error, not a genuinely high-risk case)
suspicious = df[(df["risk_label"] == "High") &
                 (df["tumor_mutation_burden"] < 5) &
                 (df["ctdna_level_ng_ml"] < 2)]
print("\nSuspicious rows found (possible false biomarker correlations):")
print(suspicious if not suspicious.empty else "None found in this mini dataset - clean!")

# In this mini dataset there are none - saved as-is for the ML Engineer.
# (In the real project, EDA Engineer would DROP or FIX confirmed false labels here.)
df.to_csv("data/eda_checked_dataset.csv", index=False)
print("\nEDA-checked dataset saved to data/eda_checked_dataset.csv")
