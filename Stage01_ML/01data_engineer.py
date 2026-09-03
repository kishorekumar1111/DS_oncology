import pandas as pd
from pathlib import Path

# ============================================================
# DATA ENGINEER
# Raw Oncology Data -> Clean Master Dataset
# ============================================================

DATA_PATH = Path("data")

# ------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------

genomic = pd.read_csv(DATA_PATH / "raw_genomic_panel.csv")
ehr = pd.read_csv(DATA_PATH / "raw_ehr_labs.csv")
outcomes = pd.read_csv(DATA_PATH / "raw_historical_outcomes.csv")

print("Raw data loaded")
print("Genomic :", genomic.shape)
print("EHR     :", ehr.shape)
print("Outcomes:", outcomes.shape)


# ------------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


genomic = clean_columns(genomic)
ehr = clean_columns(ehr)
outcomes = clean_columns(outcomes)


# ------------------------------------------------------------
# 3. STANDARDIZE PATIENT IDs
# ------------------------------------------------------------

def clean_ids(df):
    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


genomic = clean_ids(genomic)
ehr = clean_ids(ehr)
outcomes = clean_ids(outcomes)


# ------------------------------------------------------------
# 4. REMOVE DUPLICATES
# ------------------------------------------------------------

genomic = genomic.drop_duplicates()
ehr = ehr.drop_duplicates()
outcomes = outcomes.drop_duplicates()

print("\nDuplicates removed")


# ------------------------------------------------------------
# 5. CONVERT NUMERIC DATA
# ------------------------------------------------------------

genomic_numeric = [
    "tumor_mutation_burden",
    "ctdna_level_ng_ml",
    "variant_allele_fraction"
]

for col in genomic_numeric:
    genomic[col] = pd.to_numeric(
        genomic[col],
        errors="coerce"
    )


ehr_numeric = [
    "age",
    "creatinine_mg_dl",
    "alt_u_l"
]

for col in ehr_numeric:
    ehr[col] = pd.to_numeric(
        ehr[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 6. AGGREGATE GENOMIC DATA
#    Multiple records -> One patient
# ------------------------------------------------------------

genomic_patient = (
    genomic
    .groupby("patient_id")
    .agg(
        gene_mutation=(
            "gene_mutation",
            lambda x: ",".join(
                x.dropna()
                 .astype(str)
                 .str.upper()
                 .unique()
            )
        ),

        mutation_count=(
            "gene_mutation",
            lambda x: x.dropna().nunique()
        ),

        tumor_mutation_burden=(
            "tumor_mutation_burden",
            "median"
        ),

        ctdna_level_ng_ml=(
            "ctdna_level_ng_ml",
            "median"
        ),

        max_variant_allele_fraction=(
            "variant_allele_fraction",
            "max"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# 7. AGGREGATE EHR DATA
# ------------------------------------------------------------

ehr_patient = (
    ehr
    .groupby("patient_id")
    .agg(
        age=("age", "median"),
        sex=("sex", "first"),
        smoking_status=("smoking_status", "first"),
        cancer_type=("cancer_type", "first"),
        histology=("histology", "first"),
        stage=("stage", "first"),
        creatinine_mg_dl=("creatinine_mg_dl", "median"),
        alt_u_l=("alt_u_l", "median")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8. KEEP ONE OUTCOME RECORD PER PATIENT
# ------------------------------------------------------------

outcome_patient = (
    outcomes
    .drop_duplicates(subset="patient_id")
)


# ------------------------------------------------------------
# 9. MERGE THE THREE DATA SOURCES
# ------------------------------------------------------------

master = genomic_patient.merge(
    ehr_patient,
    on="patient_id",
    how="inner"
)

master = master.merge(
    outcome_patient,
    on="patient_id",
    how="inner"
)


# ------------------------------------------------------------
# 10. HANDLE MISSING NUMERIC VALUES
# ------------------------------------------------------------

numeric_columns = master.select_dtypes(
    include="number"
).columns

for col in numeric_columns:
    master[col] = master[col].fillna(
        master[col].median()
    )


# ------------------------------------------------------------
# 11. FINAL QUALITY CHECK
# ------------------------------------------------------------

master = master.drop_duplicates(
    subset="patient_id"
)

print("\n========== MASTER DATASET ==========")
print("Rows   :", master.shape[0])
print("Columns:", master.shape[1])

print("\nMissing values:")
print(master.isnull().sum())

print("\nDuplicate patient IDs:",
      master["patient_id"].duplicated().sum())


# ------------------------------------------------------------
# 12. SAVE MASTER DATASET
# ------------------------------------------------------------

output_file = DATA_PATH / "master_dataset.csv"

master.to_csv(
    output_file,
    index=False
)

print("\nMaster dataset created successfully!")
print("Saved to:", output_file)