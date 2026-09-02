"""
ROLE: KNOWLEDGE ENGINEER
JOB: "Construct the NCCN SOP knowledge base and interface tools for drug
databases & clinical trial APIs."

WHY THIS STEP EXISTS:
An agent can't reason without knowledge. This role builds the "facts" the
agent will use: Standard Operating Procedures (SOPs, modeled loosely on
NCCN-style guidance) + a live therapy/clinical-trial inventory.
"""

import pandas as pd

# STEP 1: Load the live therapy/trial inventory (like a real pharmacy/trial-matching API would return)
inventory = pd.read_csv("data/therapy_trial_inventory.csv")
print("Therapy & Clinical Trial Inventory loaded:")
print(inventory)

# STEP 2: Build a small SOP knowledge base (rules the agent must follow)
# In a real project this comes from actual NCCN guideline PDFs / hospital SOP documents.
sop_knowledge_base = {
    "HIGH_RISK_RULE": "A High toxicity-risk patient MUST receive a dose-reduced regimen "
                       "and an urgent tumor board review before treatment starts.",
    "RENAL_SAFE_RULE": "NEVER assign a nephrotoxic drug (renal_safe=False) to a patient "
                        "with creatinine above 2.5 mg/dL (severe renal impairment) - "
                        "select a renal-safe alternative even if it means a less potent regimen.",
    "TIE_BREAK_RULE": "If two High-risk patients compete for the same biomarker-matched "
                       "therapy slot, prioritize the patient with the HIGHER tumor mutation "
                       "burden (more aggressive disease), unless the RENAL_SAFE_RULE overrides it.",
    "OVERRIDE_RULE": "A human oncologist can pause or override any agent decision at any time.",
}

print("\nSOP Knowledge Base:")
for rule, text in sop_knowledge_base.items():
    print(f"  [{rule}] {text}")

# Save both for the next role (Workflow Engineer) to use
inventory.to_pickle("data/inventory.pkl")
import json
with open("data/sop_knowledge_base.json", "w") as f:
    json.dump(sop_knowledge_base, f, indent=2)
print("\nSaved inventory.pkl and sop_knowledge_base.json")
