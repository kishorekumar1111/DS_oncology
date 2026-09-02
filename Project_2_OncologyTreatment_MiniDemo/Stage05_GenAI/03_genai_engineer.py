"""
ROLE: GENAI ENGINEER
JOB: "Construct the generative scenario pipeline (GAN/VAE or LLM-based
synthetic patient generation)."

WHY THIS STEP EXISTS:
Generate NEW, realistic-looking rare-mutation patient scenarios that never
actually happened, to test the treatment engine against situations we don't
have real data for.

SIMPLIFIED FOR CLASSROOM: A real GAN/VAE/LLM pipeline is heavy. Here we
sample new values from the historical mean/spread (Gaussian sampling) -
the same underlying idea ("learn the distribution, sample new realistic
points") at a beginner-friendly scale.
"""

import pandas as pd
import numpy as np

baseline = pd.read_pickle("data/baseline_stats.pkl")
np.random.seed(1)


def generate_scenario(risk_label, n=1):
    """Samples n new synthetic patients that statistically resemble the given risk level."""
    stats = baseline.loc[risk_label]
    scenarios = []
    for _ in range(n):
        tmb = max(0, np.random.normal(stats[("tumor_mutation_burden", "mean")], stats[("tumor_mutation_burden", "std")] + 1))
        ctdna = max(0, np.random.normal(stats[("ctdna_level_ng_ml", "mean")], stats[("ctdna_level_ng_ml", "std")] + 0.5))
        creatinine = max(0, np.random.normal(stats[("creatinine_mg_dl", "mean")], stats[("creatinine_mg_dl", "std")] + 0.2))
        scenarios.append({"tumor_mutation_burden": round(tmb, 1),
                           "ctdna_level_ng_ml": round(ctdna, 1),
                           "creatinine_mg_dl": round(creatinine, 1),
                           "risk_label": risk_label})
    return scenarios


# STEP: Generate 3 new synthetic "High" risk scenarios (the rare, high-impact case)
synthetic_scenarios = generate_scenario("High", n=3)

print("Generated synthetic HIGH-RISK patient scenarios (never actually happened):")
for s in synthetic_scenarios:
    print(" ", s)

pd.DataFrame(synthetic_scenarios).to_csv("data/synthetic_scenarios.csv", index=False)
print("\nSaved to data/synthetic_scenarios.csv")
