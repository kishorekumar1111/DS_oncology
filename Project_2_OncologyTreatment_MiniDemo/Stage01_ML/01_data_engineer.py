"""
ROLE: DATA ENGINEER
JOB (from problem statement): "Stitch genomic, vital sensor, and EHR logs
into a clean, timestamped master dataset."

WHY THIS STEP EXISTS:
Raw clinical data never arrives clean. Here we have 3 separate source
files (genomic/biomarker panel, EHR lab results, historical toxicity
outcomes). Before ANY model can be trained, someone has to merge them
into ONE table and fix missing/duplicate values. If this step is skipped
or done wrong, every downstream role (and every downstream AI agent)
inherits the mistake.
"""

import pandas as pd

# STEP 1: Load the 3 raw source files (this mimics 3 different real hospital systems)
genomic_df = pd.read_csv("data/raw_genomic_panel.csv")        # mutation panel + ctDNA
labs_df = pd.read_csv("data/raw_ehr_labs.csv")                # renal/liver labs + age
outcomes_df = pd.read_csv("data/raw_historical_outcomes.csv")  # past toxicity outcome labels

print("Raw genomic panel file BEFORE cleaning:")
print(genomic_df)

# STEP 2: Fix real-world messiness
# 2a. Remove exact duplicate rows (P2 appears twice in raw_genomic_panel.csv)
genomic_df = genomic_df.drop_duplicates()

# 2b. Handle missing values (P8 has a blank alt_u_l, P10 has a blank creatinine_mg_dl)
#     Simple beginner-safe fix: fill missing numeric values with the column average
labs_df["alt_u_l"] = labs_df["alt_u_l"].fillna(labs_df["alt_u_l"].mean())
labs_df["creatinine_mg_dl"] = labs_df["creatinine_mg_dl"].fillna(labs_df["creatinine_mg_dl"].mean())

# STEP 3: Merge all 3 sources into ONE master table, matched by patient_id
# (In the real project this would also be matched by sample timestamp, not just patient_id)
master_df = genomic_df.merge(labs_df, on="patient_id").merge(outcomes_df, on="patient_id")

# STEP 4: Save the clean master dataset for the next role (EDA Engineer) to use
master_df.to_csv("data/master_dataset.csv", index=False)

print("\nCleaned MASTER dataset (saved to data/master_dataset.csv):")
print(master_df)
print(f"\nRows before cleaning: {len(genomic_df) + 1} (with duplicate) -> Rows after: {len(master_df)}")
