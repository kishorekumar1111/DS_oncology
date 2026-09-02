"""
ROLE: DATA ENGINEER (SLM stage)
JOB: "Curate and format oncology report-summary pairs into a clean
fine-tuning dataset."

WHY THIS STEP EXISTS:
A summarizer needs to be shown examples of (dense patient chart -> what an
oncologist would actually say instead). This role prepares that training
data cleanly.
"""

import pandas as pd

df = pd.read_csv("data/mini_reports.csv")

# STEP 1: Basic cleaning - remove double spaces, strip whitespace
df["full_report"] = df["full_report"].str.replace(r"\s+", " ", regex=True).str.strip()

# STEP 2: Split each report into sentences (needed by the SLM Engineer next)
df["sentences"] = df["full_report"].apply(lambda t: [s.strip() for s in t.split(". ") if s.strip()])

print(f"Prepared {len(df)} patient chart reports for summarization.")
for _, row in df.iterrows():
    print(f"\n{row['report_id']}: {len(row['sentences'])} sentences")

df.to_pickle("data/prepared_reports.pkl")
print("\nSaved to data/prepared_reports.pkl")
