"""Shared feature spec for the toxicity risk score (no outcome leakage)."""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "toxicity_event_binary"
GENE_LIST = ["ALK", "BRAF", "EGFR", "KEAP1", "KRAS", "MET", "PIK3CA", "ROS1", "STK11", "TP53"]

NUM_COLS = [
    "age",
    "mutation_count",
    "tumor_mutation_burden",
    "ctdna_level_ng_ml",
    "creatinine_mg_dl",
    "max_variant_allele_fraction",
]
BIN_COLS = ["age_missing"] + [f"gene_{g}" for g in GENE_LIST]
CAT_COLS = ["sex", "smoking_status", "histology", "treatment_name", "line_of_therapy_clean"]

LEAKAGE_COLS = {
    "pfs_months", "overall_survival_months", "best_response", "progression_status",
    "toxicity_grade", "toxicity_event", "toxicity_event_binary", "pfs_gt_os_flag",
}

DATA_CANDIDATES = [
    Path(__file__).resolve().parent / "dataset.csv",
    Path(__file__).resolve().parent / "Data" / "master_dataset_300_CLEANED.csv",
    Path(__file__).resolve().parent / "Data" / "master_dataset_300.csv",
    Path(__file__).resolve().parent / "Data" / "master_dataset.csv",
]


def resolve_dataset() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No oncology dataset found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in DATA_CANDIDATES)
    )


def _normalize_lot(series: pd.Series) -> pd.Series:
    lot_map = {
        "1": "1L", "1l": "1L", "1L": "1L", "first line": "1L", "firstline": "1L",
        "2": "2L", "2l": "2L", "2L": "2L", "second line": "2L",
    }
    key = series.astype(str).str.strip()
    mapped = key.map(lambda x: lot_map.get(x, lot_map.get(x.lower(), np.nan)))
    return mapped


def prepare_xy(df: pd.DataFrame):
    """Return model frame X, label y, abstain mask, and display metadata.

    alt_u_l is excluded (low-trust placeholder in this cohort).
    Outcome columns are excluded to prevent leakage.
    """
    d = df.copy()
    d = d.dropna(subset=[TARGET]).reset_index(drop=True)

    if "age_missing" in d.columns:
        d.loc[d["age_missing"].astype(float) == 1, "age"] = np.nan
    else:
        d["age_missing"] = d["age"].isna().astype(int) if "age" in d.columns else 0

    if "line_of_therapy_clean" not in d.columns:
        src = d["line_of_therapy"] if "line_of_therapy" in d.columns else pd.Series(np.nan, index=d.index)
        d["line_of_therapy_clean"] = _normalize_lot(src)

    gene_raw = d["gene_mutation"] if "gene_mutation" in d.columns else pd.Series("", index=d.index)
    gene_raw = gene_raw.fillna("").astype(str)
    for gene in GENE_LIST:
        d[f"gene_{gene}"] = gene_raw.apply(
            lambda s, g=gene: int(g in [x.strip().upper() for x in s.split(",") if x.strip()])
        )

    for col in NUM_COLS + BIN_COLS + CAT_COLS:
        if col not in d.columns:
            d[col] = np.nan

    hist_missing = d["histology"].isna() | d["histology"].astype(str).str.strip().isin(["", "nan", "None"])
    tx_missing = d["treatment_name"].isna() | d["treatment_name"].astype(str).str.strip().isin(["", "nan", "None"])
    age_missing = d["age_missing"].fillna(0).astype(int).eq(1) | d["age"].isna()
    abstain = (hist_missing | tx_missing | age_missing).to_numpy()

    X = d[NUM_COLS + BIN_COLS + CAT_COLS].copy()
    for c in NUM_COLS + BIN_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CAT_COLS:
        X[c] = X[c].astype(object).where(pd.notna(X[c]), np.nan)

    y = d[TARGET].astype(int)
    meta_cols = [c for c in ["patient_id", "age", "sex", "histology", "gene_mutation",
                              "treatment_name", "line_of_therapy_clean", TARGET] if c in d.columns]
    meta = d[meta_cols].copy()
    return X, y, abstain, meta
