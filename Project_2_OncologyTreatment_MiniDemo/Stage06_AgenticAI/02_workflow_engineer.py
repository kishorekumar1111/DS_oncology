"""
ROLE: WORKFLOW ENGINEER
JOB: "Deconstruct macro goals ('Optimize 2nd-Line Therapy for EGFR+ NSCLC')
into concrete, tool-executable subtasks."

WHY THIS STEP EXISTS:
An agent can't act on a vague goal like "Optimize therapy for this patient"
directly. This role breaks it into small, concrete steps the Agent
Engineer's reasoning loop can execute one at a time.
"""

def break_down_goal(patient_id, gene_mutation, risk_level):
    """Turns one macro goal into an ordered list of concrete subtasks."""
    subtasks = [f"Check {gene_mutation}-targeted therapy inventory for {patient_id}"]

    if risk_level == "High":
        subtasks.append(f"Reserve {gene_mutation}-targeted therapy for {patient_id} IMMEDIATELY")
        subtasks.append(f"Check renal/hepatic safety profile before dosing {patient_id}")
        subtasks.append(f"Escalate {patient_id} to tumor board for urgent review")
    elif risk_level == "Moderate":
        subtasks.append(f"Put standard-dose {gene_mutation}-targeted therapy on standby for {patient_id}")
        subtasks.append(f"Monitor {patient_id} labs every 2 weeks")
    else:
        subtasks.append(f"No immediate action needed for {patient_id}, continue monitoring")

    return subtasks


if __name__ == "__main__":
    # STEP: Demo breakdown of a macro goal from the mission brief
    goal_patient, goal_gene, goal_risk = "P4", "EGFR", "High"
    tasks = break_down_goal(goal_patient, goal_gene, goal_risk)

    print(f"Macro Goal: 'Optimize 2nd-Line Therapy for {goal_gene}+ NSCLC' "
          f"(patient={goal_patient}, risk={goal_risk})")
    print("Broken down into concrete subtasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    import json
    with open("data/subtasks.json", "w") as f:
        json.dump({"patient": goal_patient, "gene": goal_gene, "risk": goal_risk, "subtasks": tasks}, f, indent=2)
    print("\nSaved to data/subtasks.json")
