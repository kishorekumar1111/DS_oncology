# Stage 06 — Agentic AI (Mini Demo)

**Mission:** Bring everything together — an autonomous agent that reads
ML toxicity risk, DL pathology alerts, and NLP/SLM clinical intelligence,
then decides its own precision treatment plan and reserves therapy/trial
slots, with a transparent reasoning trace and a physician override option.

**Simplified for classroom pace:** A real agent uses an LLM with tool-calling
in a reasoning loop (the "ReAct" pattern: Reason -> Act -> Observe -> repeat).
Here we hand-code that SAME loop with simple if/else rules, so students see
the loop structure and the reasoning trace clearly, without needing an LLM API.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_knowledge_engineer.py` | Knowledge Engineer | Builds the SOP knowledge base + therapy/trial inventory |
| `02_workflow_engineer.py` | Workflow Engineer | Breaks a macro goal ("Optimize 2nd-Line Therapy for EGFR+ NSCLC") into sub-tasks |
| `03_agent_engineer.py` | Agent Engineer | Runs the reasoning loop that decides therapy reservations |
| `04_evaluation_engineer.py` | Evaluation Engineer | Tests the agent on the mission brief's hard case: high TMB + severe renal dysfunction |
| `05_integration_engineer.py` | Integration Engineer | Command-center style function with physician override control |
