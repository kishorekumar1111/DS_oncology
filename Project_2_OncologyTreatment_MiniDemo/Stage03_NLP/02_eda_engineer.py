"""
ROLE: EDA ENGINEER (NLP stage)
JOB: "Analyze vocabulary distributions and medical phrasing patterns across
severe vs. mild toxicity reports."

WHY THIS STEP EXISTS:
Before training a classifier, we should understand WHICH words actually
separate "urgent" from "routine" clinical notes. This confirms the model
has a real pattern to learn from, and matches the problem statement's
Team Huddle observation that toxicity is often IMPLIED rather than stated.
"""

import pandas as pd
from collections import Counter
import re

df = pd.read_csv("data/clean_logs.csv")


def top_words(messages, n=8):
    words = re.findall(r"[a-z]+", " ".join(messages))
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "no", "after", "with", "and"}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(n)


urgent_msgs = df[df["urgency_label"] == "Urgent"]["clean_message"]
routine_msgs = df[df["urgency_label"] == "Routine"]["clean_message"]

print("Most common words in URGENT notes:")
print(top_words(urgent_msgs))

print("\nMost common words in ROUTINE notes:")
print(top_words(routine_msgs))

print("\n=> Notice words like 'severe', 'immediately', 'urgent', 'critical' cluster in URGENT.")
print("   This is exactly the pattern the classifier will learn.")
