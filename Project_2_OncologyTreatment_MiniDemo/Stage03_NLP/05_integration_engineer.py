"""
ROLE: INTEGRATION ENGINEER (NLP stage)

JOB:
Wire the trained urgency classifier and entity extraction
into one live function that the dashboard can call.
"""

import re
import joblib


# =========================================================
# LOAD TRAINED NLP MODEL
# =========================================================

vectorizer = joblib.load("data/vectorizer.pkl")

clf = joblib.load(
    "data/urgency_classifier.pkl"
)


# =========================================================
# ENTITY EXTRACTION
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
    # Example: 80mg, 200mg, 150 mg
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
    # ADVERSE EVENT EXTRACTION
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


    return {

        "Gene Mutation": gene_mutation,

        "Drug Name": drug_name,

        "Dosage Level": dosage_level,

        "Adverse Event": adverse_event

    }


# =========================================================
# LIVE NLP PROCESSING FUNCTION
# =========================================================

def process_incoming_message(text):

    """
    This function is called by app.py.

    Step 1:
    Predict urgency using TF-IDF + Logistic Regression.

    Step 2:
    Extract gene mutation, drug name, dosage and adverse event.
    """

    urgency = clf.predict(
        vectorizer.transform(
            [text.lower()]
        )
    )[0]


    entities = extract_entities(
        text
    )


    return {

        "urgency": urgency,

        **entities

    }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=== LIVE DEMO: Incoming clinical note ==="
    )


    incoming = (
        "Critical! EGFR positive patient reports severe rash and diarrhea "
        "after 80mg osimertinib dose. Needs review now."
    )


    result = process_incoming_message(
        incoming
    )


    print(
        f"\nMessage: {incoming}\n"
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )
