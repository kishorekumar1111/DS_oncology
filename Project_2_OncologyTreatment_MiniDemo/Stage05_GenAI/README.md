# Stage 05 — Generative AI (Mini Demo)

**Mission:** Rare genomic variants and drug resistance mutations are sparse
in real data. Use Generative AI to synthesize realistic rare compound
mutation scenarios to stress-test the whole pipeline before real deployment.

**Simplified for classroom pace:** Real GenAI here would use a GAN/VAE or an
LLM prompt pipeline. Here we generate synthetic patients by sampling from
the statistical distribution (mean/spread) of the real historical genomic
data - the same core idea (learn the distribution, sample new realistic
points) at a beginner-friendly scale.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Loads historical genomic stats as the reference baseline |
| `02_eda_prompt_engineer.py` | EDA / Prompt Engineer | Finds gaps (rare combos) in historical data to target |
| `03_genai_engineer.py` | GenAI Engineer | Generates new synthetic rare-mutation patient scenarios |
| `04_evaluation_engineer.py` | Evaluation Engineer | Checks generated scenarios are realistic, not nonsense |
| `05_integration_engineer.py` | Integration Engineer | Feeds a synthetic scenario into the Stage 1 ML model as a stress test |
