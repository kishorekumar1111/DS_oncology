"""
ROLE: EDA ENGINEER (SLM stage)
JOB: "Audit token distribution to ensure specialized medical codes and
dosage figures are retained."

WHY THIS STEP EXISTS:
If summarization accidentally drops important clinical words (like "renal",
"mg", "biomarker", "tumor board"), the briefing becomes useless or even
dangerous for a responder. This role checks those key words survive.
"""

import pandas as pd
from collections import Counter
import re

df = pd.read_pickle("data/prepared_reports.pkl")

IMPORTANT_TERMS = ["renal", "hepatic", "creatinine", "ctdna", "mutation burden",
                    "dose", "mg", "tumor board", "toxicity"]

for _, row in df.iterrows():
    words = re.findall(r"[a-z]+", row["full_report"].lower())
    counts = Counter(words)
    print(f"{row['report_id']} — key term counts:")
    for term in IMPORTANT_TERMS:
        term_words = term.split()
        found = sum(1 for w in words if any(t in w for t in term_words))
        if found:
            print(f"  '{term}': appears {found} time(s) -> must be KEPT in summary")
    print()
