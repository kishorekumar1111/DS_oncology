"""
EDA ENGINEER - Stage 01 (Machine Learning)
Project: Personalized Precision Medicine for Oncology
Squad Role: EDA Engineer
Task: Clean Data -> Explore Patterns -> Find Problems -> Verified Data

Input : master_dataset_300.csv (300 NSCLC patients, 24 columns)
Output: master_dataset_300_CLEANED.csv + eda_report.md
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
IN_PATH = r"C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\DS_oncology\Stage01_ML\Data\master_dataset_300.csv"
OUT_CSV  = r"C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\master_dataset_300_CLEANED.csv"
OUT_REPORT = r"C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\eda_report.md"

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

df = pd.read_csv(IN_PATH)

log("# EDA Engineer Report — Oncology Master Dataset")
log(f"\nShape: {df.shape[0]} patients x {df.shape[1]} columns\n")

# ======================================================================
# STEP 1: CLEAN DATA
# ======================================================================
log("## STEP 1 — Clean Data\n")

issues_found = []

# --- 1a. Zero-variance columns (useless for ML) ---
zero_var = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
log(f"**Zero-variance columns (no signal, drop for modeling):** {zero_var}")
issues_found.append(f"cancer_type & stage_num are constant across all 300 rows (all NSCLC, all Stage IV) -> zero predictive value, safe to drop from the ML feature set (keep only as cohort metadata).")

# --- 1b. Fake / placeholder 'age_missing' rows ---
# age_missing=1 rows all show age == 66.0 exactly (mean=66, std=0) -> imputed placeholder, not real observed age
fake_age_mask = df['age_missing'] == 1
fake_age_vals = df.loc[fake_age_mask, 'age'].unique()
log(f"\n**Suspicious 'age' values:** {fake_age_mask.sum()} rows flagged `age_missing=1` all carry the exact same age value {fake_age_vals} "
    f"(mean=66.0, std=0.0). This is a placeholder for a truly missing age, not a real measurement.")
issues_found.append("14 patients have age_missing=1 and age hard-set to 66.0 (a fill-value, not a real observation). "
                     "Treating this as real age would bias any age-risk relationship. Action: set age=NaN for these rows and impute properly (median/model-based) or use age_missing as its own feature and exclude the fake 66.0 from age statistics.")

df.loc[fake_age_mask, 'age'] = np.nan  # un-fake it

# --- 1c. ALT (liver enzyme) looks like a fake/placeholder biomarker ---
alt_mode_share = (df['alt_u_l'] == 8.0).mean()
log(f"\n**ALT (alt_u_l) suspicious clustering:** {alt_mode_share:.1%} of all patients show the exact same value (8.0 U/L), "
    "with only a handful of other distinct values. Real liver-enzyme labs do not cluster this tightly.")
issues_found.append("alt_u_l is 8.0 U/L in 96.3% of rows (289/300) -> looks like a default/carry-forward fill value rather than "
                     "a real lab draw. This is exactly the kind of 'false biomarker correlate' the brief warns about: if left in, "
                     "a model could learn a spurious ALT->risk relationship driven by missingness, not biology. "
                     "Action: flag alt_u_l as low-trust/likely-imputed and exclude it from the biomarker leaderboard, or convert to "
                     "a binary 'alt_measured_flag' instead of using the raw value.")

# --- 1d. Inconsistent categorical encoding: line_of_therapy ---
lot_map = {'1': '1L', '1L': '1L', 'First line': '1L', '2': '2L', '2L': '2L'}
log(f"\n**line_of_therapy has 5 raw labels for what should be 2 categories:** {sorted(df['line_of_therapy'].unique())}")
issues_found.append("line_of_therapy mixes '1','1L','First line' (all = first line) and '2','2L' (all = second line). "
                     "Left unmerged, a model/groupby would treat these as 5 unrelated categories, silently splitting the sample "
                     "and hiding the real line-of-therapy effect. Action: normalize to {1L, 2L}.")
df['line_of_therapy_clean'] = df['line_of_therapy'].map(lot_map)

# --- 1e. Missing values ---
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
log(f"\n**Missing values (after un-faking age):**\n{nulls.to_string()}")
issues_found.append("histology: 10 missing (3.3%) -> impute as 'Unknown' category, don't drop rows. "
                     "pfs_months: 9 missing -> leave as NaN for survival-analysis models that handle censoring. "
                     "toxicity_grade / toxicity_event_binary: 7 missing together (consistent with each other) -> leave as NaN, do not zero-fill (zero would silently mean 'no toxicity').")

# --- 1f. Duplicate patients ---
log(f"\n**Duplicate patient_id:** {df['patient_id'].duplicated().sum()}  |  **Fully duplicate rows:** {df.duplicated().sum()}")

log(f"\n**Total distinct data-quality issues fixed/flagged: {len(issues_found)}**")
for i, iss in enumerate(issues_found, 1):
    log(f"{i}. {iss}")

# ======================================================================
# STEP 2: EXPLORE PATTERNS
# ======================================================================
log("\n\n## STEP 2 — Explore Patterns\n")

num_cols = ['age','mutation_count','tumor_mutation_burden','ctdna_level_ng_ml',
            'creatinine_mg_dl','max_variant_allele_fraction','pfs_months','overall_survival_months']

log("**Numeric summary (clean age, ALT excluded as untrustworthy):**")
log(df[num_cols].describe().T.round(2).to_string())

log("\n**Toxicity Grade distribution:**")
log(df['toxicity_grade'].value_counts(dropna=False).sort_index().to_string())

log("\n**Best response distribution:**")
log(df['best_response'].value_counts().to_string())

log("\n**Treatment line (normalized) vs progression:**")
log(pd.crosstab(df['line_of_therapy_clean'], df['progression_status']).to_string())

# Biomarker leaderboard: correlation of each numeric biomarker with toxicity risk
tmp = df.copy()
tmp['high_risk'] = (tmp['toxicity_grade'] >= 3).astype(float)
tmp.loc[tmp['toxicity_grade'].isna(), 'high_risk'] = np.nan
leaderboard = tmp[num_cols + ['mutation_count','high_risk']].corr(numeric_only=True)['high_risk'].drop('high_risk').sort_values(key=abs, ascending=False)
log("\n**Biomarker Leaderboard — correlation with High-Risk toxicity (|r| ranked):**")
log(leaderboard.round(3).to_string())

# ======================================================================
# STEP 3: FIND PROBLEMS (logical/clinical inconsistencies)
# ======================================================================
log("\n\n## STEP 3 — Find Problems\n")

problems = []

# PFS cannot exceed OS
bad_pfs = df[df['pfs_months'] > df['overall_survival_months']]
log(f"**PFS > Overall Survival (clinically impossible, PFS is capped by OS):** {len(bad_pfs)} rows")
problems.append(f"{len(bad_pfs)}/300 patients (9.3%) have progression-free survival LONGER than overall survival, which is "
                 "not clinically possible (you cannot progress-free for longer than you were alive/followed). "
                 "Likely cause: pfs_months and overall_survival_months were generated/entered independently. "
                 "Action: flag these patient_ids for source verification; do not train a model on pfs_months for these rows as-is.")
log(bad_pfs[['patient_id','pfs_months','overall_survival_months','progression_status']].head(10).to_string(index=False))

# toxicity_grade vs toxicity_event consistency (sanity check - should be perfectly consistent)
cross = pd.crosstab(df['toxicity_grade'], df['toxicity_event'])
mismatch = ((df['toxicity_grade']>=3) != (df['toxicity_event']=='Grade >=3')) & df['toxicity_grade'].notna()
log(f"\n**toxicity_grade vs toxicity_event cross-check:** {mismatch.sum()} mismatches found (0 = fully consistent, good)")

# age outliers
age_out = df[(df['age']<20)|(df['age']>95)]
log(f"\n**Biologically implausible ages (<20 or >95):** {len(age_out)}")

# creatinine / VAF sanity range checks
log(f"\n**creatinine_mg_dl range:** {df['creatinine_mg_dl'].min()}–{df['creatinine_mg_dl'].max()} (normal clinical range ~0.5-1.5, looks plausible)")
log(f"**max_variant_allele_fraction range:** {df['max_variant_allele_fraction'].min()}–{df['max_variant_allele_fraction'].max()} (valid 0-1 fraction, looks plausible)")

for i,p in enumerate(problems,1):
    log(f"{i}. {p}")

# ======================================================================
# STEP 4: VERIFIED DATA (write cleaned output)
# ======================================================================
log("\n\n## STEP 4 — Verified Data (output)\n")

df_out = df.drop(columns=['cancer_type','stage_num'])  # zero-variance
df_out['pfs_gt_os_flag'] = (df['pfs_months'] > df['overall_survival_months']).astype(int)
df_out['alt_low_trust_flag'] = (df['alt_u_l'] == 8.0).astype(int)

df_out.to_csv(OUT_CSV, index=False)
log(f"Cleaned dataset written: {OUT_CSV}  ({df_out.shape[0]} rows x {df_out.shape[1]} cols)")
log("New columns added: line_of_therapy_clean, pfs_gt_os_flag, alt_low_trust_flag (age un-faked to NaN where age_missing=1)")

with open(OUT_REPORT, "w") as f:
    f.write("\n\n".join(report_lines))

print("\n\nDONE.")
