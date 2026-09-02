import pandas as pd

# 1. Load raw datasets
genomic = pd.read_csv("data/raw_genomic_panel.csv")
ehr = pd.read_csv("data/raw_ehr_labs.csv")
outcomes = pd.read_csv("data/raw_historical_outcomes.csv")

# 2. Standardize column names
for df in [genomic, ehr, outcomes]:
    df.columns = df.columns.str.strip().str.lower()

# 3. Clean patient IDs
for df in [genomic, ehr, outcomes]:
    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

# 4. Remove duplicate records
genomic = genomic.drop_duplicates()
ehr = ehr.drop_duplicates()
outcomes = outcomes.drop_duplicates()

# 5. Convert numeric columns
for col in [
    "tumor_mutation_burden",
    "ctdna_level_ng_ml",
    "variant_allele_fraction"
]:
    genomic[col] = pd.to_numeric(genomic[col], errors="coerce")

for col in ["age", "creatinine_mg_dl", "alt_u_l"]:
    ehr[col] = pd.to_numeric(ehr[col], errors="coerce")

# 6. Combine multiple genomic records for each patient
genomic_patient = genomic.groupby("patient_id").agg({
    "gene_mutation": lambda x: ",".join(x.dropna().astype(str).unique()),
    "tumor_mutation_burden": "median",
    "ctdna_level_ng_ml": "median",
    "variant_allele_fraction": "max"
}).reset_index()

# 7. Keep one EHR record per patient
ehr_patient = ehr.groupby("patient_id").agg({
    "age": "median",
    "sex": "first",
    "smoking_status": "first",
    "cancer_type": "first",
    "histology": "first",
    "stage": "first",
    "creatinine_mg_dl": "median",
    "alt_u_l": "median"
}).reset_index()

# 8. Keep one outcome record per patient
outcome_patient = outcomes.drop_duplicates("patient_id")

# 9. Merge all sources
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

# 10. Handle missing numerical values
numeric_cols = master.select_dtypes(
    include="number"
).columns

for col in numeric_cols:
    master[col] = master[col].fillna(master[col].median())

# 11. Save master dataset
master.to_csv(
    "data/master_dataset.csv",
    index=False
)

print("Master dataset created successfully!")
print("Rows:", len(master))
print("Columns:", len(master.columns))
print(master.head())