#!/usr/bin/env python3
"""Collect ablation results: scan output/neurostorm_ablation/*.log for test metric."""
import csv
import re
from pathlib import Path

OUT_BASE = Path("output/neurostorm_ablation")

# Each entry: (run_name, group, task)
RUNS = [
    ("A_sex",  "A_stride_reg",  "sex"),
    ("A_age",  "A_stride_reg",  "age"),
    ("B1_sex", "B_TPT_p10",     "sex"),
    ("B1_age", "B_TPT_p10",     "age"),
    ("B2_sex", "B_TPT_p30",     "sex"),
    ("B2_age", "B_TPT_p30",     "age"),
    ("B3_sex", "B_TPT_p50",     "sex"),
    ("B3_age", "B_TPT_p50",     "age"),
    ("C_sex",  "C_augment",     "sex"),
    ("C_age",  "C_augment",     "age"),
    ("B4_sex", "B_TPT_p120",    "sex"),
    ("B4_age", "B_TPT_p120",    "age"),
    ("B5_sex", "B_TPT_p160",    "sex"),
    ("B5_age", "B_TPT_p160",    "age"),
]


def read_metric(log_path: Path, task: str):
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="ignore")
    key = "test_acc" if task == "sex" else "test_mae"
    # Try table format first: │ test_acc │ 0.9123 │
    m = re.findall(rf"{key}\s+│\s+([0-9.eE+-]+)", text)
    if not m:
        m = re.findall(rf"{key}[^0-9-]*([0-9.]+)", text)
    return float(m[-1]) if m else None


def main():
    rows = []
    for name, group, task in RUNS:
        log = OUT_BASE / f"{name}.log"
        done = (OUT_BASE / name / "_done").exists()
        metric = read_metric(log, task)
        rows.append((group, task, name, "DONE" if done else "RUNNING/FAIL", metric))

    print(f"{'group':<16} {'task':<5} {'name':<8} {'status':<14} metric")
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<5} {r[2]:<8} {r[3]:<14} {r[4]}")

    csv_path = OUT_BASE / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "task", "run", "status", "metric"])
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Best per task
    print("\n=== Best per task (lower MAE / higher acc) ===")
    for task in ("sex", "age"):
        cands = [r for r in rows if r[1] == task and r[4] is not None]
        if not cands:
            continue
        if task == "sex":
            best = max(cands, key=lambda r: r[4])
        else:
            best = min(cands, key=lambda r: r[4])
        print(f"  {task}: {best[2]} ({best[0]}) -> {best[4]}")


if __name__ == "__main__":
    main()
