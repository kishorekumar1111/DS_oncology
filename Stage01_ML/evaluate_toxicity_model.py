"""Independent evaluation of the locked toxicity risk pipeline."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from toxicity_features import RANDOM_STATE, TARGET, prepare_xy, resolve_dataset

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def main():
    artifact = joblib.load(BASE_DIR / "toxicity_risk_model.pkl")
    pipeline = artifact["pipeline"]
    model_name = artifact["model_name"]
    threshold = float(artifact["decision_threshold"])
    intended = artifact.get("intended_use", "")

    print(f"Loaded: {model_name}")
    print(f"Frozen threshold: {threshold:.2f}")
    print(intended)

    df = pd.read_csv(resolve_dataset())
    X, y, abstain, meta = prepare_xy(df)
    X_tr, X_te, y_tr, y_te, ab_tr, ab_te, i_tr, i_te = train_test_split(
        X, y, abstain, np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    proba = pipeline.predict_proba(X_te)[:, 1]
    pred_frozen = (proba >= threshold).astype(int)
    pred_default = (proba >= 0.5).astype(int)

    print(f"\nHeld-out test: {len(y_te)} patients ({y_te.sum()} toxic / {(y_te==0).sum()} non-toxic)")
    print(f"Abstain flags on test: {int(ab_te.sum())}")

    metrics = {
        "ROC-AUC": roc_auc_score(y_te, proba),
        "PR-AUC (Average Precision)": average_precision_score(y_te, proba),
        "Brier Score (lower=better)": brier_score_loss(y_te, proba),
        "Accuracy @ frozen": accuracy_score(y_te, pred_frozen),
        "Precision (PPV) @ frozen": precision_score(y_te, pred_frozen, zero_division=0),
        "Recall / Sensitivity @ frozen": recall_score(y_te, pred_frozen, zero_division=0),
        "Specificity @ frozen": specificity_score(y_te, pred_frozen),
        "F1 @ frozen": f1_score(y_te, pred_frozen, zero_division=0),
        "MCC @ frozen": matthews_corrcoef(y_te, pred_frozen),
        "Cohen's Kappa @ frozen": cohen_kappa_score(y_te, pred_frozen),
    }
    print("\n=== METRICS @ FROZEN THRESHOLD (selected on train OOF) ===")
    for k, v in metrics.items():
        print(f"  {k:40s}: {v:.3f}")

    sweep_rows = []
    for t in np.arange(0.05, 0.95, 0.05):
        p = (proba >= t).astype(int)
        sweep_rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(y_te, p, zero_division=0),
                "recall": recall_score(y_te, p, zero_division=0),
                "specificity": specificity_score(y_te, p),
                "f1": f1_score(y_te, p, zero_division=0),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    cm1 = confusion_matrix(y_te, pred_default)
    cm2 = confusion_matrix(y_te, pred_frozen)
    for ax, cm, title in (
        (axes[0, 0], cm1, "Confusion @ 0.50 (not the operating point)"),
        (axes[0, 1], cm2, f"Confusion @ frozen {threshold:.2f}"),
    ):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low/No Tox", "Severe Tox"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Low/No Tox", "Severe Tox"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    axes[1, 0].plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    axes[1, 0].plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    axes[1, 0].plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    axes[1, 0].axvline(threshold, color="gray", ls="--", label=f"frozen ({threshold:.2f})")
    axes[1, 0].set_title("Test-set sweep (do not retune on this)")
    axes[1, 0].legend(fontsize=8)

    try:
        frac, meanp = calibration_curve(y_te, proba, n_bins=5, strategy="quantile")
        axes[1, 1].plot(meanp, frac, marker="o", label=model_name)
    except Exception:
        axes[1, 1].scatter(proba, y_te, alpha=0.4)
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 1].set_title("Calibration")
    axes[1, 1].set_xlabel("Mean predicted risk")
    axes[1, 1].set_ylabel("Observed rate")
    axes[1, 1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "evaluation_report.png", dpi=150)
    print("Saved: evaluation_report.png")

    test_meta = meta.iloc[i_te].copy()
    test_meta["y_true"] = y_te.values
    test_meta["proba"] = proba
    subgroup_rows = []
    for col in [c for c in ["histology", "treatment_name", "sex", "line_of_therapy_clean"] if c in test_meta.columns]:
        for val, grp in test_meta.groupby(col):
            if grp["y_true"].nunique() < 2 or len(grp) < 5:
                auc = "insufficient data"
            else:
                auc = round(roc_auc_score(grp["y_true"], grp["proba"]), 3)
            subgroup_rows.append(
                {
                    "subgroup_type": col,
                    "subgroup_value": val,
                    "n": len(grp),
                    "positive_rate": round(grp["y_true"].mean(), 3),
                    "roc_auc": auc,
                }
            )
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(OUT_DIR / "subgroup_fairness_analysis.csv", index=False)

    error_df = test_meta.copy()
    error_df["predicted_risk_pct"] = (proba * 100).round(1)
    error_df["predicted_label_frozen"] = pred_frozen
    error_df["abstain"] = ab_te.astype(int)

    def err_type(row):
        if row["abstain"]:
            return "ABSTAIN"
        a, p = row["y_true"], row["predicted_label_frozen"]
        if a == 1 and p == 0:
            return "FALSE NEGATIVE (missed toxic patient)"
        if a == 0 and p == 1:
            return "FALSE POSITIVE (over-flagged)"
        if a == 1 and p == 1:
            return "TRUE POSITIVE"
        return "TRUE NEGATIVE"

    error_df["error_category"] = error_df.apply(err_type, axis=1)
    error_df.to_csv(OUT_DIR / "error_analysis_detailed.csv", index=False)
    fn = error_df[error_df["error_category"].str.contains("FALSE NEGATIVE")]
    fp = error_df[error_df["error_category"].str.contains("FALSE POSITIVE")]
    print(f"\nFN={len(fn)}  FP={len(fp)}  @ frozen threshold {threshold:.2f}")

    auc = metrics["ROC-AUC"]
    if auc >= 0.75:
        verdict = "STRONG discrimination on this split — still requires external validation."
    elif auc >= 0.60:
        verdict = "MODERATE signal — workflow-ready prototype only."
    else:
        verdict = (
            "WEAK / NEAR-RANDOM on available features. Locked pipeline is a process demo, "
            "not a clinical classifier."
        )

    with open(OUT_DIR / "evaluation_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("EVALUATION ENGINEER — LOCKED PIPELINE EVALUATION\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Frozen threshold: {threshold:.2f} ({artifact.get('threshold_rule', '')})\n")
        f.write(f"Test set: {len(y_te)} patients ({int(y_te.sum())} toxic)\n")
        f.write(f"{intended}\n\n")
        f.write("1. METRICS @ FROZEN THRESHOLD\n" + "-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:40s}: {v:.3f}\n")
        f.write(f"\n2. ERRORS @ FROZEN THRESHOLD\n" + "-" * 40 + "\n")
        f.write(f"  False negatives: {len(fn)}\n  False positives: {len(fp)}\n")
        f.write(f"\n3. SUBGROUPS (small-n; interpret cautiously)\n" + "-" * 40 + "\n")
        f.write(subgroup_df.to_string(index=False) if not subgroup_df.empty else "  n/a\n")
        f.write(f"\n\n4. VERDICT\n" + "-" * 40 + "\n")
        f.write(f"  ROC-AUC = {auc:.3f} -> {verdict}\n")

    print("Saved: evaluation_summary_report.txt")
    print("\nEVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
