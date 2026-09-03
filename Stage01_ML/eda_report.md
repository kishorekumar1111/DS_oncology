# EDA Engineer Report — Oncology Master Dataset


Shape: 300 patients x 24 columns


## STEP 1 — Clean Data


**Zero-variance columns (no signal, drop for modeling):** ['cancer_type', 'stage_num']


**Suspicious 'age' values:** 14 rows flagged `age_missing=1` all carry the exact same age value [66.] (mean=66.0, std=0.0). This is a placeholder for a truly missing age, not a real measurement.


**ALT (alt_u_l) suspicious clustering:** 96.3% of all patients show the exact same value (8.0 U/L), with only a handful of other distinct values. Real liver-enzyme labs do not cluster this tightly.


**line_of_therapy has 5 raw labels for what should be 2 categories:** ['1', '1L', '2', '2L', 'First line']


**Missing values (after un-faking age):**
age                      14
histology                10
pfs_months                9
toxicity_grade            7
toxicity_event_binary     7


**Duplicate patient_id:** 0  |  **Fully duplicate rows:** 0


**Total distinct data-quality issues fixed/flagged: 5**

1. cancer_type & stage_num are constant across all 300 rows (all NSCLC, all Stage IV) -> zero predictive value, safe to drop from the ML feature set (keep only as cohort metadata).

2. 14 patients have age_missing=1 and age hard-set to 66.0 (a fill-value, not a real observation). Treating this as real age would bias any age-risk relationship. Action: set age=NaN for these rows and impute properly (median/model-based) or use age_missing as its own feature and exclude the fake 66.0 from age statistics.

3. alt_u_l is 8.0 U/L in 96.3% of rows (289/300) -> looks like a default/carry-forward fill value rather than a real lab draw. This is exactly the kind of 'false biomarker correlate' the brief warns about: if left in, a model could learn a spurious ALT->risk relationship driven by missingness, not biology. Action: flag alt_u_l as low-trust/likely-imputed and exclude it from the biomarker leaderboard, or convert to a binary 'alt_measured_flag' instead of using the raw value.

4. line_of_therapy mixes '1','1L','First line' (all = first line) and '2','2L' (all = second line). Left unmerged, a model/groupby would treat these as 5 unrelated categories, silently splitting the sample and hiding the real line-of-therapy effect. Action: normalize to {1L, 2L}.

5. histology: 10 missing (3.3%) -> impute as 'Unknown' category, don't drop rows. pfs_months: 9 missing -> leave as NaN for survival-analysis models that handle censoring. toxicity_grade / toxicity_event_binary: 7 missing together (consistent with each other) -> leave as NaN, do not zero-fill (zero would silently mean 'no toxicity').



## STEP 2 — Explore Patterns


**Numeric summary (clean age, ALT excluded as untrustworthy):**

                             count   mean    std    min    25%    50%    75%    max
age                          286.0  65.63  11.43  35.00  58.00  66.00  73.75  90.00
mutation_count               300.0   1.75   0.70   1.00   1.00   2.00   2.00   3.00
tumor_mutation_burden        300.0   6.62   5.90   0.10   2.59   5.19   8.46  40.00
ctdna_level_ng_ml            300.0   5.22   4.80   0.21   1.91   3.50   7.46  32.47
creatinine_mg_dl             300.0   1.01   0.23   0.55   0.85   1.03   1.18   1.65
max_variant_allele_fraction  300.0   0.44   0.15   0.07   0.35   0.46   0.57   0.65
pfs_months                   291.0   9.36   5.35   1.00   5.55   7.90  12.20  31.90
overall_survival_months      300.0  22.34  10.07   4.00  15.18  20.85  27.12  70.00


**Toxicity Grade distribution:**

toxicity_grade
0.0    90
1.0    65
2.0    70
3.0    45
4.0    23
NaN     7


**Best response distribution:**

best_response
PR    128
SD     89
PD     72
CR     11


**Treatment line (normalized) vs progression:**

progression_status     Not progressed  Progressed
line_of_therapy_clean                            
1L                                 70         106
2L                                 53          71


**Biomarker Leaderboard — correlation with High-Risk toxicity (|r| ranked):**

creatinine_mg_dl               0.081
max_variant_allele_fraction   -0.077
ctdna_level_ng_ml              0.065
pfs_months                    -0.036
overall_survival_months        0.030
age                           -0.028
mutation_count                 0.025
mutation_count                 0.025
tumor_mutation_burden         -0.008



## STEP 3 — Find Problems


**PFS > Overall Survival (clinically impossible, PFS is capped by OS):** 28 rows

patient_id  pfs_months  overall_survival_months progression_status
     P0012        16.5                     15.3         Progressed
     P0024        26.1                      5.4     Not progressed
     P0028        18.3                     18.2         Progressed
     P0041        13.2                     13.0     Not progressed
     P0043        25.9                      9.9         Progressed
     P0044        17.8                     11.8         Progressed
     P0051        14.6                     12.1     Not progressed
     P0057        27.4                     14.9         Progressed
     P0076        18.3                     13.4     Not progressed
     P0079        12.2                     10.3         Progressed


**toxicity_grade vs toxicity_event cross-check:** 0 mismatches found (0 = fully consistent, good)


**Biologically implausible ages (<20 or >95):** 0


**creatinine_mg_dl range:** 0.55–1.65 (normal clinical range ~0.5-1.5, looks plausible)

**max_variant_allele_fraction range:** 0.073–0.649 (valid 0-1 fraction, looks plausible)

1. 28/300 patients (9.3%) have progression-free survival LONGER than overall survival, which is not clinically possible (you cannot progress-free for longer than you were alive/followed). Likely cause: pfs_months and overall_survival_months were generated/entered independently. Action: flag these patient_ids for source verification; do not train a model on pfs_months for these rows as-is.



## STEP 4 — Verified Data (output)


Cleaned dataset written: C:\Users\HP\OneDrive\Desktop\ONCOLOGY EDA\master_dataset_300_CLEANED.csv  (300 rows x 25 cols)

New columns added: line_of_therapy_clean, pfs_gt_os_flag, alt_low_trust_flag (age un-faked to NaN where age_missing=1)