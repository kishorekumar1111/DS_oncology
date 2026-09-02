"""
ROLE: NLP ENGINEER
JOB: Construct the text classification pipeline and medical entity
extraction NER models.

TWO JOBS:
1. Classify each clinical note as Urgent/Routine
2. Extract Gene Mutation, Drug Name, Dosage Level and Adverse Event

CLASSROOM VERSION:
TF-IDF + Logistic Regression for urgency classification.
Rule-based (dictionary + regex) entity extraction as a simple NER stand-in.
"""

import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/clean_logs.csv")


# =========================================================
# PART A — URGENCY TEXT CLASSIFIER
# =========================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["clean_message"]
)

y = df["urgency_label"]


clf = LogisticRegression()

clf.fit(X, y)


# Save trained NLP models

joblib.dump(
    vectorizer,
    "data/vectorizer.pkl"
)

joblib.dump(
    clf,
    "data/urgency_classifier.pkl"
)


print(
    f"Urgency classifier trained. "
    f"Training accuracy: {clf.score(X, y):.0%}"
)


# =========================================================
# PART B — RULE-BASED NER
# =========================================================

GENE_MUTATIONS = ["egfr", "kras", "alk", "ros1", "braf", "met"]

DRUG_NAMES = [
    "osimertinib", "erlotinib", "pembrolizumab", "carboplatin",
    "crizotinib", "dabrafenib"
]

ADVERSE_EVENTS = [
    "rash", "diarrhea", "pneumonitis", "neutropenia", "hepatotoxicity",
    "fever", "nausea", "fatigue", "dehydration"
]


def extract_entities(text):

    text_lower = text.lower()

    # -----------------------------------------------------
    # GENE MUTATION
    # -----------------------------------------------------

    gene_found = [
        gene.upper()
        for gene in GENE_MUTATIONS
        if gene in text_lower
    ]

    gene_mutation = (
        ", ".join(gene_found)
        if gene_found
        else "Not found"
    )


    # -----------------------------------------------------
    # DRUG NAME
    # -----------------------------------------------------

    drug_found = [
        drug.title()
        for drug in DRUG_NAMES
        if drug in text_lower
    ]

    drug_name = (
        ", ".join(drug_found)
        if drug_found
        else "Not found"
    )


    # -----------------------------------------------------
    # DOSAGE LEVEL
    # Example: "80mg", "200mg", "150 mg"
    # -----------------------------------------------------

    dosage_match = re.search(
        r"(\d+)\s*mg",
        text_lower
    )

    dosage_level = (
        f"{dosage_match.group(1)}mg"
        if dosage_match
        else "Not found"
    )


    # -----------------------------------------------------
    # ADVERSE EVENT
    # -----------------------------------------------------

    found_events = [
        word
        for word in ADVERSE_EVENTS
        if word in text_lower
    ]

    if found_events:

        adverse_event = ", ".join(
            found_events
        )

    else:

        adverse_event = "None"


    # -----------------------------------------------------
    # RETURN NER RESULTS
    # -----------------------------------------------------

    return {

        "Gene Mutation": gene_mutation,

        "Drug Name": drug_name,

        "Dosage Level": dosage_level,

        "Adverse Event": adverse_event

    }


# =========================================================
# SAMPLE NER TEST
# =========================================================

print(
    "\nSample NER extraction on 2 messages:"
)


for msg in df["message"].head(2):

    print(
        f"  Message: {msg}"
    )

    print(
        f"  Extracted: {extract_entities(msg)}\n"
    )
