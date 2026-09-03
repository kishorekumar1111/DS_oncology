"""
Patient toxicity risk score — locked, leak-free clinical prototype.

Champion is chosen on training-fold PR-AUC only (logistic vs shallow HGB),
then calibrated. The decision threshold is frozen from out-of-fold scores.
Held-out patients are scored once; training patients are not published as
"real-time" risk. Intended use: research / tumor-board support, not care.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from toxicity_features import (
    BIN_COLS,
    CAT_COLS,
    NUM_COLS,
    RANDOM_STATE,
    prepare_xy,
    resolve_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR
TARGET_RECALL = 0.70
INTENDED_USE = (
    "RESEARCH / TUMOR-BOARD SUPPORT ONLY. Not a diagnostic device. "
    "Do not use for treatment decisions without independent clinical validation."
)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), BIN_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CAT_COLS,
            ),
        ]
    )


def make_logistic():
    return LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def make_hgb():
    return HistGradientBoostingClassifier(
        max_depth=3,
        min_samples_leaf=20,
        learning_rate=0.05,
        max_iter=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def oof_pr_auc(estimator, X, y, cv):
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return average_precision_score(y, proba), roc_auc_score(y, proba), proba


def choose_threshold(y_true, proba, target_recall=TARGET_RECALL, min_specificity=0.10):
    """Lowest threshold that hits target recall with usable specificity; else Youden."""
    grid = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append((t, rec, spec, rec + spec - 1))
    table = pd.DataFrame(rows, columns=["threshold", "recall", "specificity", "youden"])
    hits = table[(table["recall"] >= target_recall) & (table["specificity"] >= min_specificity)]
    if len(hits):
        chosen = hits.iloc[0]
        rule = f"recall>={target_recall:.2f} and specificity>={min_specificity:.2f}"
    else:
        chosen = table.loc[table["youden"].idxmax()]
        rule = (
            "youden_max (recall target not reachable without collapsing specificity)"
        )
    return float(chosen["threshold"]), rule, table


def risk_tier(p, threshold, abstain_flag):
    if abstain_flag:
        return "ABSTAIN"
    if p >= max(threshold, 0.5):
        return "HIGH"
    if p >= threshold:
        return "MEDIUM"
    return "LOW"


def evaluate_split(name, y, proba, threshold, abstain):
    y = np.asarray(y)
    proba = np.asarray(proba)
    usable = ~np.asarray(abstain, dtype=bool)
    if usable.sum() < 5 or len(np.unique(y[usable])) < 2:
        usable = np.ones(len(y), dtype=bool)
    y_u, p_u = y[usable], proba[usable]
    pred = (p_u >= threshold).astype(int)
    metrics = {
        "split": name,
        "n": int(usable.sum()),
        "n_abstain": int((~usable).sum()) if abstain is not None else 0,
        "roc_auc": roc_auc_score(y_u, p_u),
        "pr_auc": average_precision_score(y_u, p_u),
        "brier": brier_score_loss(y_u, p_u),
        "recall_at_threshold": recall_score(y_u, pred, zero_division=0),
        "threshold": threshold,
    }
    cm = confusion_matrix(y_u, pred)
    print(f"\n--- {name} (threshold={threshold:.2f}) ---")
    for k, v in metrics.items():
        if k in ("split",):
            continue
        print(f"  {k:22s}: {v:.3f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    print("  confusion [TN FP; FN TP]:", cm.tolist())
    print(classification_report(y_u, pred, target_names=["Low/No Toxicity", "Severe Toxicity"], zero_division=0))
    return metrics, pred, usable


def inner_estimator(calibrated_pipe):
    cal = calibrated_pipe.named_steps["clf"]
    inner = cal.calibrated_classifiers_[0]
    return getattr(inner, "estimator", getattr(inner, "base_estimator", None))


def main():
    data_path = resolve_dataset()
    df = pd.read_csv(data_path)
    X, y, abstain, meta = prepare_xy(df)
    print(f"Loaded {data_path.name}: {len(X)} labeled patients, {X.shape[1]} raw feature columns")
    print(f"Positive rate: {y.mean()*100:.1f}%  |  abstain candidates: {abstain.sum()}")

    idx = np.arange(len(X))
    X_tr, X_te, y_tr, y_te, ab_tr, ab_te, i_tr, i_te = train_test_split(
        X, y, abstain, idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = [("LogisticRegression", make_logistic()), ("HistGradientBoosting", make_hgb())]
    oof_store = {}
    print("\nTraining-fold model comparison (out-of-fold, not test):")
    best_name, best_est, best_pr, best_oof = None, None, -1, None
    for name, est in candidates:
        pr, roc, oof = oof_pr_auc(est, X_tr, y_tr, cv)
        oof_store[name] = (pr, roc, oof)
        print(f"  {name:22s}  OOF PR-AUC={pr:.3f}  OOF ROC-AUC={roc:.3f}")
        if pr > best_pr:
            best_name, best_est, best_pr, best_oof = name, est, pr, oof

    # Prefer logistic on ties or near-ties: more stable probabilities on small N.
    log_pr = oof_store["LogisticRegression"][0]
    if best_name != "LogisticRegression" and (best_pr - log_pr) < 0.01:
        best_name, best_est, best_pr, best_oof = (
            "LogisticRegression",
            make_logistic(),
            log_pr,
            oof_store["LogisticRegression"][2],
        )
        print("  -> near-tie: keeping LogisticRegression (simpler, better calibrated on small N)")

    print(f"\nChampion (CV PR-AUC): {best_name}")

    threshold, thresh_rule, thresh_table = choose_threshold(y_tr, best_oof)
    print(f"Frozen decision threshold: {threshold:.2f}  ({thresh_rule})")

    calibrated = Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                CalibratedClassifierCV(
                    best_est,
                    method="sigmoid",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                ),
            ),
        ]
    )
    calibrated.fit(X_tr, y_tr)

    te_proba = calibrated.predict_proba(X_te)[:, 1]
    te_metrics, _, _ = evaluate_split(
        "HELD-OUT TEST", y_te, te_proba, threshold, np.zeros(len(y_te), dtype=bool)
    )

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fpr, tpr, _ = roc_curve(y_te, te_proba)
    axes[0, 0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"{best_name} AUC={te_metrics['roc_auc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC — held-out test")
    axes[0, 0].legend()

    prec, rec, _ = precision_recall_curve(y_te, te_proba)
    axes[0, 1].plot(rec, prec, color="#DD8452", lw=2, label=f"PR-AUC={te_metrics['pr_auc']:.3f}")
    axes[0, 1].axhline(y_te.mean(), color="k", ls="--", alpha=0.4, label="prevalence")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall — held-out test")
    axes[0, 1].legend()

    try:
        frac, meanp = calibration_curve(y_te, te_proba, n_bins=5, strategy="quantile")
        axes[1, 0].plot(meanp, frac, marker="o", label=best_name)
    except Exception:
        axes[1, 0].scatter(te_proba, y_te, alpha=0.4)
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 0].set_xlabel("Mean predicted risk")
    axes[1, 0].set_ylabel("Observed event rate")
    axes[1, 0].set_title("Calibration — held-out test")

    fitted = inner_estimator(calibrated)
    feat_names = calibrated.named_steps["prep"].get_feature_names_out()
    if hasattr(fitted, "coef_"):
        imp = pd.Series(fitted.coef_.ravel(), index=feat_names).abs().sort_values(ascending=False).head(12)
        xlabel = "|logistic coefficient|"
    else:
        imp = pd.Series(fitted.feature_importances_, index=feat_names).sort_values(ascending=False).head(12)
        xlabel = "importance"
    axes[1, 1].barh(imp.index[::-1], imp.values[::-1], color="#55A868")
    axes[1, 1].set_title(f"Top drivers — {best_name}")
    axes[1, 1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_performance_report.png", dpi=150)
    print("Saved: model_performance_report.png")

    # Held-out scores only
    scored = meta.iloc[i_te].copy()
    scored["toxicity_risk_score_pct"] = (te_proba * 100).round(1)
    scored["risk_tier"] = [risk_tier(p, threshold, a) for p, a in zip(te_proba, ab_te)]
    scored["abstain"] = ab_te.astype(int)
    scored = scored.sort_values("toxicity_risk_score_pct", ascending=False)
    scored.to_csv(OUT_DIR / "patient_toxicity_risk_scores.csv", index=False)
    print("Saved: patient_toxicity_risk_scores.csv (held-out patients only)")
    print(scored["risk_tier"].value_counts().to_string())

    artifact = {
        "pipeline": calibrated,
        "model_name": f"Calibrated {best_name}",
        "decision_threshold": threshold,
        "threshold_rule": thresh_rule,
        "intended_use": INTENDED_USE,
        "num_cols": NUM_COLS,
        "bin_cols": BIN_COLS,
        "cat_cols": CAT_COLS,
        "cv_pr_auc": float(best_pr),
        "cv_roc_auc": float(oof_store[best_name][1]),
        "test_metrics": te_metrics,
    }
    joblib.dump(artifact, OUT_DIR / "toxicity_risk_model.pkl")
    print("Saved: toxicity_risk_model.pkl")

    with open(OUT_DIR / "model_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("TOXICITY RISK SCORE - LOCKED PIPELINE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"INTENDED USE: {INTENDED_USE}\n\n")
        f.write(f"Labeled cohort: {len(X)} patients  |  event rate {y.mean()*100:.1f}%\n")
        f.write(f"Champion (selected on train OOF PR-AUC): {best_name}\n")
        f.write(f"Train OOF PR-AUC={best_pr:.3f}  ROC-AUC={oof_store[best_name][1]:.3f}\n")
        f.write(f"Frozen threshold: {threshold:.2f} ({thresh_rule})\n\n")
        f.write("Held-out test (not used for model or threshold selection):\n")
        f.write(
            f"  ROC-AUC={te_metrics['roc_auc']:.3f}  PR-AUC={te_metrics['pr_auc']:.3f}  "
            f"Brier={te_metrics['brier']:.3f}  Recall@thr={te_metrics['recall_at_threshold']:.3f}\n"
        )
        f.write("\nComparison (train OOF):\n")
        for name, (pr, roc, _) in oof_store.items():
            f.write(f"  {name:22s} PR-AUC={pr:.3f}  ROC-AUC={roc:.3f}\n")
        f.write("\nTop |weights| / importances:\n")
        for feat, val in imp.items():
            f.write(f"  {feat}: {val:.4f}\n")
        f.write("\nHeld-out risk tiers:\n")
        f.write(scored["risk_tier"].value_counts().to_string())
        f.write("\n")
        if te_metrics["roc_auc"] < 0.60:
            f.write(
                "\nNOTE: Discrimination is weak on this dataset. Treat scores as a "
                "workflow demo, not a validated clinical classifier.\n"
            )
    print("Saved: model_summary_report.txt")
    print("\nML PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
