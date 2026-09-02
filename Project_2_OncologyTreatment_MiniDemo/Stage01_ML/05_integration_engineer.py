"""
ROLE: INTEGRATION ENGINEER
JOB (from problem statement): "Wrap the model in an ultra-fast API for
real-time clinical dashboard consumption."

WHY THIS STEP EXISTS:
A trained model sitting in a notebook is useless to a real tumor board.
This role makes it USABLE - normally via a real web API (e.g. Flask/FastAPI)
that a dashboard can call. Here we simulate that with a simple Python
function that behaves exactly like an API: takes an input, returns a
clean JSON-style response.

(In the real large-scale project, this would be a proper Flask/FastAPI
server running continuously. For a 1-hour classroom demo we simulate the
same INPUT -> OUTPUT behaviour without needing a running server.)
"""

import joblib
import pandas as pd

model = joblib.load("data/risk_model.pkl")
features = ["tumor_mutation_burden", "ctdna_level_ng_ml", "creatinine_mg_dl", "alt_u_l"]


def get_patient_risk_score(tumor_mutation_burden, ctdna_level_ng_ml, creatinine_mg_dl, alt_u_l):
    """
    This function IS the API endpoint.
    In a real API this would be called like:  POST /predict-toxicity-risk
    """
    input_row = pd.DataFrame([[tumor_mutation_burden, ctdna_level_ng_ml, creatinine_mg_dl, alt_u_l]],
                              columns=features)
    prediction = model.predict(input_row)[0]
    confidence = model.predict_proba(input_row).max()

    # This is the "Clinical Field Briefing Sheet" style response - simple, non-technical
    response = {
        "risk_level": prediction,
        "confidence": f"{confidence:.0%}",
        "recommendation": {
            "Low": "Standard-dose regimen. Routine monitoring.",
            "Moderate": "Consider dose adjustment. Increase lab monitoring frequency.",
            "High": "Reduce starting dose and escalate to tumor board IMMEDIATELY.",
        }[prediction],
    }
    return response


# STEP: Demo call, exactly like the 8:00 AM tumor board scenario from the mission brief
if __name__ == "__main__":
    print("=== LIVE DEMO: New patient biomarker panel comes in ===")
    result = get_patient_risk_score(
        tumor_mutation_burden=13.8, ctdna_level_ng_ml=7.1, creatinine_mg_dl=1.9, alt_u_l=68
    )
    print("API Response:")
    for key, value in result.items():
        print(f"  {key}: {value}")
