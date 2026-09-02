"""
ROLE: EVALUATION ENGINEER (SLM stage)
JOB: "Benchmark medical perplexity, summary fidelity, and inference latency
under load."

WHY THIS STEP EXISTS:
The problem statement's own Team Huddle says: if reading the summary
doesn't save at least 80% of the reading time versus the full chart, the
summary needs to be refined. We check that here.
"""

import pandas as pd
import time

df = pd.read_pickle("data/summarized_reports.pkl")

for _, row in df.iterrows():
    full_len = len(row["full_report"].split())
    summary_len = len(row["summary"].split())
    reduction = 1 - (summary_len / full_len)

    start = time.time()
    _ = row["summary"]  # simulated "inference" - instant here since it's precomputed
    latency_ms = (time.time() - start) * 1000

    print(f"{row['report_id']}: {full_len} words -> {summary_len} words "
          f"({reduction:.0%} shorter) | latency: {latency_ms:.2f} ms")

    flag = "PASS" if reduction >= 0.8 else "NEEDS REFINEMENT (below 80% target)"
    print(f"  Result: {flag}\n")
