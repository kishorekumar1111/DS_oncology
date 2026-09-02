# Stage 01 — Machine Learning (Mini Demo)

**Mission (from the problem statement):** Turn raw genomic/biomarker and EHR
lab data (tumor mutation burden, ctDNA level, renal/liver function) into an
instant treatment toxicity/response risk score for a patient: Low / Moderate / High.

**Algorithm used:** Decision Tree Classifier (simple, explainable — matches the
study material's "classification fundamentals, decision trees" requirement,
and is easy to defend at a real tumor board).

## Run order (= the 5 squad roles, in the exact order they work in real life)

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Merges 3 raw source files into one clean master table |
| `02_eda_engineer.py` | EDA Engineer | Plots patterns, finds and removes a false biomarker correlation |
| `03_ml_engineer.py` | ML Engineer | Trains the Decision Tree toxicity risk classifier |
| `04_evaluation_engineer.py` | Evaluation Engineer | Tests the model on unseen data, checks for overconfidence |
| `05_integration_engineer.py` | Integration Engineer | Wraps the trained model in a simple "API" function |

Run them in this exact order — each script reads the file the previous script created.
