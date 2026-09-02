"""
ROLE: INTEGRATION ENGINEER (SLM stage)
JOB: "Package the SLM into an offline tactical briefing mobile/desktop app
for clinical rounds."

WHY THIS STEP EXISTS:
Wraps the summarizer into ONE simple function that works with zero
internet/cloud dependency - exactly what an oncologist's device needs at
the bedside.
"""

import re

IMPORTANT_TERMS = {"renal", "hepatic", "creatinine", "ctdna", "mutation", "dose",
                    "toxicity", "critical", "urgent", "immediate", "elevated",
                    "reduction", "tumor", "board"}


def score_sentence(sentence):
    words = re.findall(r"[a-z]+", sentence.lower())
    return sum(1 for w in words if any(term in w for term in IMPORTANT_TERMS))


def get_field_briefing(full_report, keep=2):
    """This IS the offline briefing app's core function - no internet needed."""
    sentences = [s.strip() for s in full_report.split(". ") if s.strip()]
    scored = sorted(sentences, key=score_sentence, reverse=True)[:keep]
    ordered = [s for s in sentences if s in scored]
    return ". ".join(ordered) + "."


if __name__ == "__main__":
    incoming_report = (
        "Patient P6 is a 68-year-old with KRAS-mutant NSCLC and high toxicity risk. "
        "Tumor mutation burden is 15.7 mut/Mb, among the highest in the cohort. "
        "ctDNA level is 8.9 ng/mL, indicating significant disease burden. "
        "Creatinine is elevated at 2.1 mg/dL, consistent with renal impairment. "
        "ALT is elevated at 75 U/L. Patient is a critical case requiring immediate "
        "dose reduction and close renal monitoring. Recommend urgent tumor board "
        "review before next treatment cycle."
    )
    print("=== LIVE DEMO: Offline 2-line oncologist briefing ===")
    print("FULL REPORT:\n", incoming_report)
    print("\nBRIEFING:\n", get_field_briefing(incoming_report))
