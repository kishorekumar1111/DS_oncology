# Stage 03 — Natural Language Processing (Mini Demo)

**Mission:** Extract structure from unstructured physician/nurse clinical notes -
classify urgency, and pull out Gene Mutation / Drug Name / Dosage Level / Adverse Event.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Loads + cleans raw clinical text logs |
| `02_eda_engineer.py` | EDA Engineer | Compares word usage in urgent vs routine notes |
| `03_nlp_engineer.py` | NLP Engineer | Trains an urgency text classifier + a simple rule-based NER |
| `04_evaluation_engineer.py` | Evaluation Engineer | Checks precision/recall, inspects wrong predictions |
| `05_integration_engineer.py` | Integration Engineer | Live intake function: note text in -> tagged entities + urgency out |
