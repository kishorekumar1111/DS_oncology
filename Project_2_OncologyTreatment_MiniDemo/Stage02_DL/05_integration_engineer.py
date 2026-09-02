import numpy as np
import joblib

vision_model = joblib.load("data/vision_model.pkl")
seq_params = joblib.load("data/sequence_model.pkl")


def get_pathology_alert(
    image_flat,
    current_month,
    current_ctdna,
    ml_risk_level="Unknown"
):

    vision_call = vision_model.predict([image_flat])[0]
    vision_call = str(vision_call).strip()

    predicted_ctdna_3mo = (
        seq_params["intercept"]
        + seq_params["slope"] * (current_month + 3)
    )

    predicted_ctdna_3mo = float(predicted_ctdna_3mo)

    ml_risk = str(ml_risk_level).strip().lower()

    pathology_malignant = (
        vision_call.lower() == "malignant"
    )

    ctdna_high = predicted_ctdna_3mo > 7.0
    ctdna_watch = predicted_ctdna_3mo > 5.0

    ml_high = ml_risk in [
        "high",
        "very high",
        "critical"
    ]

    ml_medium = ml_risk in [
        "medium",
        "moderate"
    ]

    if (
        ml_high
        or (pathology_malignant and ctdna_high)
        or (pathology_malignant and ml_medium)
    ):
        unified_alert = "CRITICAL"

    elif (
        pathology_malignant
        or ctdna_watch
        or ml_medium
    ):
        unified_alert = "WATCH"

    else:
        unified_alert = "STABLE"

    return {
        "pathology_says": vision_call,
        "predicted_ctdna_in_3mo": round(predicted_ctdna_3mo, 2),
        "unified_alert": unified_alert,
        "ml_risk_level": ml_risk_level,
        "pathology_signal": (
            "POSITIVE"
            if pathology_malignant
            else "NEGATIVE"
        ),
        "ctdna_signal": (
            "HIGH"
            if ctdna_high
            else "WATCH"
            if ctdna_watch
            else "NORMAL"
        )
    }


if __name__ == "__main__":

    test_image = np.load("data/X_test.npy")[0]

    result = get_pathology_alert(
        test_image,
        current_month=6,
        current_ctdna=6.2,
        ml_risk_level="High"
    )

    print("=== LIVE DEMO: Unified Pathology + ctDNA + ML Alert ===")

    for key, value in result.items():
        print(f"  {key}: {value}")
