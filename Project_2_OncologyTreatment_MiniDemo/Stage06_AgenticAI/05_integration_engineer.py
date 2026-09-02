"""
ROLE: INTEGRATION ENGINEER (Agentic AI stage)
JOB: "Build the Oncology Command Center UI featuring live agent reasoning
monitoring and manual override."

WHY THIS STEP EXISTS:
Wraps the whole agent loop into ONE command-center style function, with a
human override switch - matching the problem statement's requirement for
a "Physician Override Control".
"""

import pandas as pd
import json


def run_command_center(patient_id, gene_mutation, risk_level, subtasks_fn, agent_fn, inventory, human_override=False):
    """This IS the command center's core function."""
    print(f"=== COMMAND CENTER: Incoming case for {patient_id} (gene={gene_mutation}, risk={risk_level}) ===")

    if human_override:
        print("[HUMAN OVERRIDE ACTIVE] Agent paused. Awaiting manual oncologist decision.")
        return {"status": "PAUSED_FOR_HUMAN", "patient_id": patient_id}

    subtasks = subtasks_fn(patient_id, gene_mutation, risk_level)
    print(f"Plan generated: {len(subtasks)} subtasks")

    trace = agent_fn(subtasks, inventory)
    print("\nLive reasoning trace:")
    for line in trace:
        print(f"  {line}")

    return {"status": "COMPLETED", "patient_id": patient_id, "steps_executed": len(trace)}


if __name__ == "__main__":
    ns = {}
    exec(open("02_workflow_engineer.py").read().split("if __name__")[0], ns)
    exec(open("03_agent_engineer.py").read().split("if __name__")[0], ns)

    inventory = pd.read_pickle("data/inventory.pkl")

    print("\n--- DEMO 1: Normal run ---")
    result1 = run_command_center("P4", "EGFR", "High", ns["break_down_goal"], ns["run_agent"],
                                  inventory.copy(), human_override=False)
    print("\nResult:", result1)

    print("\n--- DEMO 2: Human override active ---")
    result2 = run_command_center("P6", "KRAS", "High", ns["break_down_goal"], ns["run_agent"],
                                  inventory.copy(), human_override=True)
    print("\nResult:", result2)
