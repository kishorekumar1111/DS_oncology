"""
ROLE: AGENT ENGINEER
JOB: "Design the core reasoning loop, tool invocation parameters, and
decision logic."

WHY THIS STEP EXISTS:
This is the heart of Agentic AI - a REASON -> ACT -> OBSERVE loop (the
"ReAct" pattern mentioned in the study material). The agent looks at the
situation, decides an action, "does" it (calls a tool), observes the
result, and repeats until the goal is done.

SIMPLIFIED FOR CLASSROOM: A real agent uses an LLM to generate each
reasoning step. Here we hand-code the same LOOP STRUCTURE with simple
rules, so students see exactly how the loop works.
"""

import re
import pandas as pd
import json

inventory = pd.read_pickle("data/inventory.pkl")
with open("data/subtasks.json") as f:
    plan = json.load(f)


def run_agent(subtasks, inventory):
    """The REASON -> ACT -> OBSERVE loop."""
    trace = []
    for step_num, task in enumerate(subtasks, 1):
        # REASON: decide what this task requires
        trace.append(f"[REASON] Step {step_num}: Need to '{task}'")

        # ACT: call a "tool" (here: check/update the therapy inventory dataframe)
        if "reserve" in task.lower() and "therapy" in task.lower():
            # Try to detect which biomarker this task is targeting
            gene_match = re.search(r"\b(EGFR|KRAS|ALK|ROS1|BRAF|MET)\b", task.upper())

            candidates = inventory[(inventory["resource_type"] == "Therapy") &
                                    (inventory["status"] == "Available")]

            if gene_match:
                matched = candidates[candidates["biomarker_target"] == gene_match.group(1)]
                if not matched.empty:
                    candidates = matched

            if not candidates.empty:
                chosen = candidates.iloc[0]
                inventory.loc[inventory["resource_id"] == chosen["resource_id"], "status"] = "Reserved"
                trace.append(f"[ACT] Tool call: reserve_therapy({chosen['resource_id']})")
                trace.append(f"[OBSERVE] {chosen['drug_name']} ({chosen['resource_id']}) successfully reserved.")
            else:
                trace.append("[ACT] Tool call: reserve_therapy() -> FAILED")
                trace.append("[OBSERVE] No matching biomarker therapy available! Escalate to human tumor board.")
        else:
            trace.append(f"[ACT] Tool call: log_action('{task}')")
            trace.append("[OBSERVE] Logged successfully.")

    return trace


if __name__ == "__main__":
    print(f"=== Agent running plan for {plan['patient']} (gene={plan['gene']}, risk={plan['risk']}) ===\n")
    reasoning_trace = run_agent(plan["subtasks"], inventory)
    for line in reasoning_trace:
        print(line)

    inventory.to_pickle("data/inventory_after.pkl")
    with open("data/reasoning_trace.json", "w") as f:
        json.dump(reasoning_trace, f, indent=2)
    print("\nSaved full reasoning trace to data/reasoning_trace.json")
