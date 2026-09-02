"""
ROLE: EDA / PROMPT ENGINEER
JOB: "Identify blind spots in genomic data and craft prompts for extreme
drug-resistance scenarios."

WHY THIS STEP EXISTS:
Historical data alone won't include RARE, extreme biomarker combinations
(e.g. very high tumor mutation burden AND severe renal impairment happening
together). This role identifies which combinations are missing/rare, so the
GenAI Engineer knows what to specifically target when generating synthetic
patients.
"""

import pandas as pd

df = pd.read_pickle("data/historical_df.pkl")

# STEP 1: Find the observed range of each signal
print("Observed ranges in historical data:")
for col in ["tumor_mutation_burden", "ctdna_level_ng_ml", "creatinine_mg_dl"]:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

# STEP 2: Define a "blind spot" - a compound extreme not yet seen
# e.g. tumor mutation burden > max observed AND creatinine > max observed together
max_tmb = df["tumor_mutation_burden"].max()
max_creatinine = df["creatinine_mg_dl"].max()

print(f"\nBLIND SPOT identified: no historical case has tumor mutation burden > {max_tmb} mut/Mb "
      f"AND creatinine > {max_creatinine} mg/dL happening TOGETHER.")
print("This is exactly the kind of dual-driver mutation + organ-impairment scenario the "
      "GenAI Engineer should generate next, to stress-test the pipeline.")

# A "prompt" for a real LLM-based generator would look like this:
prompt = (f"Generate a rare patient scenario with tumor mutation burden above {max_tmb} mut/Mb, "
          f"creatinine above {max_creatinine} mg/dL, combined with acquired drug resistance.")
print(f"\nExample prompt for a real GenAI system:\n  '{prompt}'")
