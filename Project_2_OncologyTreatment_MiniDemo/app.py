import streamlit as st
import os
import runpy
from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Oncology Tumor Board Command Center",
    page_icon="🧬",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent


# =========================================================
# LOAD STAGE SCRIPTS
# =========================================================

def load_script(rel):

    old = os.getcwd()
    folder = ROOT / Path(rel).parent

    try:
        os.chdir(folder)

        return runpy.run_path(
            str(ROOT / rel),
            run_name="dashboard_module"
        )

    finally:
        os.chdir(old)


@st.cache_resource
def load_stage_functions():

    # Stage 01 - Machine Learning
    s1 = load_script(
        "Stage01_ML/05_integration_engineer.py"
    )

    # Stage 03 - NLP
    s3 = load_script(
        "Stage03_NLP/05_integration_engineer.py"
    )

    # Stage 04 - SLM
    s4 = load_script(
        "Stage04_SLM/05_integration_engineer.py"
    )

    # Stage 05 - Generative AI
    s5 = load_script(
        "Stage05_GenAI/03_genai_engineer.py"
    )

    # Stage 06 - Agentic AI
    w = load_script(
        "Stage06_AgenticAI/02_workflow_engineer.py"
    )

    a = load_script(
        "Stage06_AgenticAI/03_agent_engineer.py"
    )

    return (
        s1["get_patient_risk_score"],
        s3["process_incoming_message"],
        s4["get_field_briefing"],
        s5["generate_scenario"],
        w["break_down_goal"],
        a["run_agent"]
    )


# =========================================================
# STAGE 02 - DEEP LEARNING
# =========================================================

def stage2_alert():

    folder = ROOT / "Stage02_DL"
    old = os.getcwd()

    try:

        os.chdir(folder)

        ns = runpy.run_path(
            str(folder / "05_integration_engineer.py"),
            run_name="dashboard_module"
        )

        test_image = np.load(
            "data/X_test.npy"
        )[0]

        return ns["get_pathology_alert"](
            test_image,
            current_month=6,
            current_ctdna=6.2
        )

    finally:

        os.chdir(old)


# =========================================================
# LOAD THERAPY / TRIAL INVENTORY
# =========================================================

def load_inventory():

    p = ROOT / "Stage06_AgenticAI/data/inventory.pkl"

    if p.exists():

        return pd.read_pickle(p)

    return pd.read_csv(
        ROOT / "Stage06_AgenticAI/data/therapy_trial_inventory.csv"
    )


# =========================================================
# AUTOMATIC CLINICAL NOTE GENERATOR
# =========================================================

def generate_clinical_note(risk_level, gene_mutation, patient_id):

    risk = risk_level.lower()

    if risk == "low":

        return (
            f"Patient {patient_id} ({gene_mutation}-mutant) remains stable on current "
            f"regimen. No new adverse events reported. Continue routine monitoring."
        )

    elif risk in ["medium", "moderate"]:

        return (
            f"Patient {patient_id} ({gene_mutation}-mutant) reports mild fatigue after "
            f"current dose. Labs trending upward. Consider closer monitoring and possible "
            f"dose review."
        )

    else:

        return (
            f"Urgent! Patient {patient_id} ({gene_mutation}-mutant) reports severe rash "
            f"and diarrhea after 80mg osimertinib dose. Grade 3 toxicity suspected, "
            f"needs review now."
        )


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.title("🧬 Oncology Tumor Board Command Center")

st.caption(
    "End-to-End Multi-Agent AI System | "
    "ML → DL → NLP → SLM → GenAI → Agentic AI"
)


# =========================================================
# SIDEBAR - PATIENT INPUT
# =========================================================

with st.sidebar:

    st.header("🧬 Patient Case Input")

    patient_id = st.selectbox(
        "Patient ID",
        [f"P{i}" for i in range(1, 11)],
        index=3
    )

    gene_mutation = st.selectbox(
        "Gene Mutation",
        ["EGFR", "KRAS", "ALK", "ROS1", "BRAF", "MET"],
        index=0
    )

    tmb = st.number_input(
        "Tumor Mutation Burden (mut/Mb)",
        0.0,
        40.0,
        10.0
    )

    ctdna = st.number_input(
        "ctDNA Level (ng/mL)",
        0.0,
        30.0,
        3.0
    )

    creatinine = st.number_input(
        "Creatinine (mg/dL)",
        0.0,
        6.0,
        1.0
    )

    alt = st.number_input(
        "ALT (U/L)",
        0.0,
        150.0,
        25.0
    )

    override = st.toggle(
        "Enable Physician Override",
        value=False
    )

    activate = st.button(
        "🧬 ACTIVATE AI TUMOR BOARD",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# WAIT UNTIL USER ACTIVATES SYSTEM
# =========================================================

if not activate:

    st.info(
        "👈 Enter a patient case in the left panel "
        "and click **ACTIVATE AI TUMOR BOARD**."
    )

    st.markdown("### System Flow")

    st.code(
        "Patient Input → ML Risk → DL Pathology → "
        "NLP Intelligence → SLM Briefing → "
        "GenAI Scenario → Agent Dispatch → Final Treatment Plan"
    )

    st.stop()


# =========================================================
# LOAD ALL AI STAGES
# =========================================================

try:

    (
        risk_fn,
        nlp_fn,
        brief_fn,
        gen_fn,
        workflow_fn,
        agent_fn

    ) = load_stage_functions()


    # Stage 01 Prediction

    risk = risk_fn(
        tmb,
        ctdna,
        creatinine,
        alt
    )


except FileNotFoundError as e:

    st.error(
        "A generated model/data file is missing. "
        "Run the stage scripts first. Missing file: "
        + str(e)
    )

    st.stop()


except Exception as e:

    st.exception(e)

    st.stop()


st.success(
    "✅ Case received. All AI stages are now connected."
)


# =========================================================
# STAGE 01 - MACHINE LEARNING
# =========================================================

st.header(
    "🟢 Stage 01 — Machine Learning: Patient Toxicity Risk Score"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Risk Level",
    risk["risk_level"]
)

c2.metric(
    "Confidence",
    risk["confidence"]
)

c3.metric(
    "Patient",
    patient_id
)

st.info(
    "📋 Recommendation: "
    + risk["recommendation"]
)


# =========================================================
# AUTOMATIC AI-GENERATED CLINICAL NOTE
# =========================================================

message = generate_clinical_note(
    risk["risk_level"],
    gene_mutation,
    patient_id
)

st.header(
    "📨 AI-Generated Clinical Note"
)

st.write(message)


# =========================================================
# STAGE 02 - DEEP LEARNING
# =========================================================

st.header(
    "🔵 Stage 02 — Deep Learning: Pathology & ctDNA Trend Alert"
)

try:

    dl = stage2_alert()

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Pathology Analysis",
        dl["pathology_says"]
    )

    d2.metric(
        "Predicted ctDNA in 3mo",
        f"{dl['predicted_ctdna_in_3mo']} ng/mL"
    )

    d3.metric(
        "Unified Alert",
        dl["unified_alert"]
    )


except Exception as e:

    st.warning(
        "Stage 02 dashboard data could not be loaded: "
        + str(e)
    )

    dl = {
        "pathology_says": "Unavailable",
        "predicted_ctdna_in_3mo": "Unavailable",
        "unified_alert": "Unavailable"
    }


# =========================================================
# STAGE 03 - NLP
# =========================================================

st.header(
    "🟠 Stage 03 — NLP: Clinical Note Intelligence"
)


# Process automatically generated clinical note

nlp = nlp_fn(message)


# ---------------------------------------------------------
# STUDENT-FRIENDLY DISPLAY
# Internal NLP = Routine
# Dashboard Display = Normal
# ---------------------------------------------------------

urgency_display = (
    "Normal"
    if nlp["urgency"].lower() == "routine"
    else nlp["urgency"]
)


st.write(
    "**Automatically Generated Clinical Note:**",
    message
)


n1, n2, n3, n4 = st.columns(4)


n1.metric(
    "Urgency",
    urgency_display
)


n2.metric(
    "Gene Mutation",
    nlp["Gene Mutation"]
)


n3.metric(
    "Dosage Level",
    nlp["Dosage Level"]
)


n4.metric(
    "Adverse Event",
    nlp["Adverse Event"]
)


# =========================================================
# STAGE 04 - SLM
# =========================================================

st.header(
    "🟣 Stage 04 — SLM: Oncologist Field Briefing"
)


full_report = (

    f"Patient {patient_id} is a {gene_mutation}-mutant NSCLC case with "
    f"{risk['risk_level']} toxicity risk. "

    f"Tumor mutation burden is {tmb} mut/Mb and "

    f"ctDNA level is {ctdna} ng/mL. "

    f"Creatinine is {creatinine} mg/dL "

    f"with ALT at {alt} U/L. "

    f"Incoming clinical note: {message}. "

    f"Stage 02 pathology result is "
    f"{dl['pathology_says']}. "

    f"Predicted ctDNA in 3 months is "
    f"{dl['predicted_ctdna_in_3mo']}. "

    f"Immediate recommendation is: "
    f"{risk['recommendation']}"
)


brief = brief_fn(
    full_report
)


st.warning(
    "🎙️ **5-Second Tumor Board Briefing:** "
    + brief
)


# =========================================================
# STAGE 05 - GENERATIVE AI
# =========================================================

st.header(
    "🟡 Stage 05 — Generative AI: Synthetic Rare-Mutation Stress Test"
)


try:

    scenarios = gen_fn(
        risk["risk_level"],
        n=3
    )


    for i, scenario in enumerate(
        scenarios,
        1
    ):

        with st.expander(
            f"🧬 Synthetic Rare-Mutation Scenario {i}",
            expanded=(i == 1)
        ):

            for key, value in scenario.items():

                st.write(
                    f"**{key.replace('_', ' ').title()}:** {value}"
                )


    st.caption(
        "Generated scenarios are used to stress-test "
        "the precision oncology treatment system."
    )


except Exception as e:

    st.warning(
        "Synthetic scenario generation unavailable: "
        + str(e)
    )


# =========================================================
# STAGE 06 - AGENTIC AI
# =========================================================

st.header(
    "🔴 Stage 06 — Agentic AI: Autonomous Treatment Engine"
)


inventory = load_inventory().copy()


if override:

    st.error(
        "⏸️ PHYSICIAN OVERRIDE ACTIVE — "
        "Agent is paused. Oncologist approval is required."
    )

    st.write({
        "status": "PAUSED_FOR_HUMAN",
        "patient_id": patient_id
    })


    col1, col2, col3 = st.columns(3)


    col1.button(
        "✅ Approve Plan",
        disabled=True
    )


    col2.button(
        "✏️ Modify Plan",
        disabled=True
    )


    col3.button(
        "❌ Cancel Dispatch",
        disabled=True
    )


    final_priority = "Paused"


else:

    # Convert reported adverse events into a simple severity flag

    has_adverse_event = (
        nlp["Adverse Event"] != "None"
    )


    # -----------------------------------------------------
    # AGENT DECISION
    #
    # High priority if:
    # NLP says Urgent OR an adverse event was reported
    # -----------------------------------------------------

    human_emergency = (

        nlp["urgency"].lower() == "urgent"

        or

        has_adverse_event
    )


    if human_emergency:

        final_priority = "HIGH"

        st.metric(
            "Final Agent Priority",
            final_priority
        )

        st.metric(
            "Adverse Event Reported",
            nlp["Adverse Event"]
        )


        st.warning(
            "⚠️ MULTI-SOURCE SAFETY SIGNAL DETECTED\n\n"
            f"ML reports {risk['risk_level']} biomarker-based risk, "
            f"but NLP detected {nlp['urgency']} urgency with "
            f"adverse event(s): {nlp['Adverse Event']}. "
            "The agent prioritizes the safety signal."
        )


        subtasks = [

            f"Assess immediate safety signal for {patient_id}",

            f"Hold current regimen and review dosage "
            f"({nlp['Dosage Level']}) for {patient_id}",

            f"Check renal/hepatic safety profile before "
            f"dosing {patient_id}",

            f"Reserve {gene_mutation}-targeted therapy alternative "
            f"for {patient_id}",

            f"Escalate {patient_id} to tumor board for urgent review",

            f"Monitor {patient_id} biomarkers closely"

        ]


    else:

        # No immediate safety signal
        # Agent follows ML risk

        final_priority = risk["risk_level"]

        st.metric(
            "Final Agent Priority",
            final_priority
        )

        st.metric(
            "Adverse Event Reported",
            nlp["Adverse Event"]
        )


        st.info(
            "No immediate safety signal was detected. "
            "The agent follows the ML risk assessment."
        )


        subtasks = workflow_fn(
            patient_id,
            gene_mutation,
            risk["risk_level"]
        )


    # -----------------------------------------------------
    # DISPLAY TREATMENT PLAN
    # -----------------------------------------------------

    trace = agent_fn(
        subtasks,
        inventory
    )


    st.subheader(
        "🤖 Autonomous Treatment Plan"
    )


    for i, task in enumerate(
        subtasks,
        1
    ):

        st.write(
            f"{i}. {task}"
        )


    # -----------------------------------------------------
    # REASONING TRACE
    # -----------------------------------------------------

    st.subheader(
        "🧠 Transparent Agent Reasoning Trace"
    )


    for line in trace:

        st.code(
            line
        )


    st.success(
        f"💊 TREATMENT PLAN COMPLETED — "
        f"{len(trace)} reasoning/action events executed "
        f"for {patient_id}."
    )


# =========================================================
# FINAL OUTCOME
# =========================================================

st.divider()


st.header(
    "🏁 FINAL OUTCOME"
)


# Student-friendly final urgency display

final_urgency_display = (
    "Normal"
    if nlp["urgency"].lower() == "routine"
    else nlp["urgency"]
)


st.markdown(
    f"""
### 🧬 Treatment Decision Summary

**Patient:** {patient_id} ({gene_mutation}-mutant)

**ML Risk:** {risk['risk_level']}
({risk['confidence']} confidence)

**NLP Urgency:** {final_urgency_display}

**Adverse Event(s):** {nlp['Adverse Event']}

**Drug / Dosage Mentioned:** {nlp['Drug Name']} / {nlp['Dosage Level']}

**Final Agent Priority:** {final_priority}

**Agent Status:** {'PAUSED FOR PHYSICIAN OVERRIDE' if override else 'AUTONOMOUS TREATMENT PLAN COMPLETED'}
"""
)


st.caption(
    "This dashboard is the visual integration layer that connects "
    "all six AI stages into one final capstone demonstration."
)
