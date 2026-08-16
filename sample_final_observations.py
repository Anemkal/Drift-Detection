"""
sample_final_observations.py

Pulls a handful of real trajectories from SWE-smith-trajectories and prints
the LAST FEW observation strings from each — resolved and unresolved
separately — so we can see what "test passed" / "test failed" actually
looks like in this dataset's real text, instead of guessing at a regex.

Run this locally, same folder as before:
    python sample_final_observations.py
"""
import json
from datasets import load_dataset

N_EACH = 6          # how many resolved / unresolved examples to sample
TAIL_STEPS = 3       # how many of the last steps to print per trajectory


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def get_observations(messages):
    """Return list of observation texts in order."""
    obs_list = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            obs_list.append(extract_text(m.get("content")))
    return obs_list


def main():
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool", streaming=True)
    ds = ds.shuffle(seed=7, buffer_size=2000)

    resolved_shown = 0
    unresolved_shown = 0

    for row in ds:
        if resolved_shown >= N_EACH and unresolved_shown >= N_EACH:
            break

        resolved = bool(row["resolved"])
        if resolved and resolved_shown >= N_EACH:
            continue
        if (not resolved) and unresolved_shown >= N_EACH:
            continue

        try:
            messages = json.loads(row["messages"])
        except (json.JSONDecodeError, TypeError):
            continue

        obs_list = get_observations(messages)
        if len(obs_list) < 2:
            continue

        label = "RESOLVED=True" if resolved else "RESOLVED=False"
        print("=" * 80)
        print(f"instance: {row['instance_id']}   [{label}]")
        print(f"total observations in trajectory: {len(obs_list)}")
        print("-" * 80)
        for i, obs in enumerate(obs_list[-TAIL_STEPS:]):
            print(f"  [last-{TAIL_STEPS - i}] {obs[:400]!r}")
        print()

        if resolved:
            resolved_shown += 1
        else:
            unresolved_shown += 1


if __name__ == "__main__":
    main()
