REFERENCE-GROUNDED SYNTHETIC NSCLC RAW DATA — 300 PATIENTS
===============================================================
Status: SYNTHETIC / EDUCATIONAL ONLY. These are not real patients.

Primary reference:
- Uploaded file: nsclc_ctdx_msk_2022_clinical_data.tsv
- The MSK NSCLC ctDNA study reports genomic and clinical data via cBioPortal.
- This package uses the uploaded reference to anchor available clinical distributions
  (for example age/TMB/clinical categories where identifiable).
- ctDNA, laboratory and toxicity variables not present in the uploaded clinical file
  are synthetic additions; they are NOT copied from real patients.

Raw sources:
01 clinical      300 rows
02 genomics      multiple mutation rows + duplicates
03 ctDNA         repeated measurements + missing values
04 labs          repeated measurements + missing values + unit strings
05 treatment     treatment/regimen + line
06 outcomes      response/progression/PFS/OS/toxicity

Intentional data-engineering issues:
- duplicate records
- missing values
- lower-case patient IDs in some records
- repeated samples/visits
- inconsistent date formats
- inconsistent category labels
- mixed numeric/unit formatting

Suggested target:
toxicity_event = Grade >=3 vs None/Grade <3

IMPORTANT:
Do not use PFS/OS/toxicity outcomes as input features when predicting the target;
that would cause target leakage. The dataset is for demonstrating data engineering
and ML workflow, not clinical validation.
