# Stage 04 — Small Language Model (Mini Demo)

**Mission:** Condense a dense patient chart into a 2-sentence, voice-ready
oncologist briefing that works fully offline.

**Simplified for classroom pace:** Real SLM fine-tuning (LoRA/PEFT on a small
transformer) needs GPU time and big libraries. Here we use extractive
summarization (picking the most important existing sentences by clinical
word-score) as a stand-in that teaches the same INPUT (long chart) -> OUTPUT
(short briefing) workflow instantly, on a laptop, offline.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Formats report-summary training pairs |
| `02_eda_engineer.py` | EDA Engineer | Checks that key clinical terms are kept, not lost |
| `03_slm_engineer.py` | SLM Engineer | "Fine-tunes" a mini summarizer (word-score extractive method) |
| `04_evaluation_engineer.py` | Evaluation Engineer | Measures how much shorter the summary is (compression %) |
| `05_integration_engineer.py` | Integration Engineer | Offline function: long chart in -> 2-line briefing out |
