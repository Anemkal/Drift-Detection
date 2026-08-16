"""
check_task_types.py

Inspects the instance_id naming pattern in SWE-smith-trajectories to see
what task types exist (e.g. func_pm_ctrl_shuffle, combine_file, pr_NNNN),
so we can decide how to cluster/split the data by task type before
re-running NSP-Monitor separately on each cluster.

Run locally:
    python check_task_types.py
"""
import re
from collections import Counter
from datasets import load_dataset

N_SAMPLE = 1000

def extract_task_type(instance_id):
    """
    Instance IDs look like:
      django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u
      lincolnloop__python-qrcode.456b01d4.combine_file__svo6amwn
      Project-MONAI__MONAI.a09c1f08.pr_2696

    The task type is the segment after the repo+hash, before the final
    __<random_id> (if present) or trailing number.
    """
    parts = instance_id.split(".")
    if len(parts) < 3:
        return "unknown"
    tail = parts[-1]  # e.g. "func_pm_ctrl_shuffle__viqnyl9u" or "pr_2696"
    # strip trailing __xxxxxxxx random suffix if present
    tail = re.sub(r'__[a-z0-9]{6,}$', '', tail)
    # strip trailing _NNNN pr numbers into a generic "pr" bucket
    tail = re.sub(r'^pr_\d+$', 'pr', tail)
    return tail


def main():
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool", streaming=True)
    ds = ds.shuffle(seed=7, buffer_size=2000)

    counts = Counter()
    counts_by_type_and_label = Counter()
    n = 0
    for row in ds:
        if n >= N_SAMPLE:
            break
        task_type = extract_task_type(row["instance_id"])
        counts[task_type] += 1
        counts_by_type_and_label[(task_type, row["resolved"])] += 1
        n += 1

    print(f"Sampled {n} instances.\n")
    print("Task type distribution:")
    for t, c in counts.most_common():
        print(f"  {t:<30} {c:>5}")

    print("\nResolved/unresolved breakdown per task type:")
    for t, _ in counts.most_common():
        n_true = counts_by_type_and_label.get((t, True), 0)
        n_false = counts_by_type_and_label.get((t, False), 0)
        print(f"  {t:<30} resolved={n_true:<5} unresolved={n_false:<5}")


if __name__ == "__main__":
    main()
