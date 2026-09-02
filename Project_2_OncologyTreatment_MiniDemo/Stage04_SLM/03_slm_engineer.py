"""
ROLE: SLM ENGINEER
JOB: "Fine-tune the small language model using parameter-efficient tuning
techniques (PEFT/LoRA)."

WHY THIS STEP EXISTS:
Turn a dense multi-sentence patient chart into 2 crisp, actionable clinical
sentences.

SIMPLIFIED FOR CLASSROOM: Real fine-tuning (LoRA/PEFT on a transformer)
needs GPU time. Here we use an extractive method: score every sentence by
how many "important" clinical words it has, then keep the TOP 2 sentences.
Same INPUT -> OUTPUT shape as a real fine-tuned SLM, runs instantly offline.
"""

import pandas as pd
import re

df = pd.read_pickle("data/prepared_reports.pkl")

IMPORTANT_TERMS = {"renal", "hepatic", "creatinine", "ctdna", "mutation", "dose",
                    "toxicity", "critical", "urgent", "immediate", "elevated",
                    "reduction", "tumor", "board"}


def score_sentence(sentence):
    words = re.findall(r"[a-z]+", sentence.lower())
    return sum(1 for w in words if any(term in w for term in IMPORTANT_TERMS))


def summarize(sentences, keep=2):
    scored = [(s, score_sentence(s)) for s in sentences]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:keep]
    # keep original order for readability
    top_sentences = [s for s in sentences if s in [t[0] for t in top]]
    return ". ".join(top_sentences) + "."


summaries = []
for _, row in df.iterrows():
    summary = summarize(row["sentences"])
    summaries.append(summary)
    print(f"{row['report_id']} SUMMARY:\n  {summary}\n")

df["summary"] = summaries
df.to_pickle("data/summarized_reports.pkl")
print("Saved summaries to data/summarized_reports.pkl")
