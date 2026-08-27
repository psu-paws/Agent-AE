"""Simulator wall time against the deployment it predicts.

The fair comparison is the simulated workload's wall time on real hardware
against the simulator's own wall time. Model load and server startup are
excluded from the measured side because the simulator has no equivalent.

Also reports what the simulations cost together, since they are independent and
run in parallel, which is the point of using a simulator for a sweep.
"""
import glob, json, os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GT = os.path.join(ROOT, "data", "ground_truth")

def real_runs():
    out = {}
    for f in glob.glob(f"{GT}/*/run_summary.json"):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        out[j["label"]] = j["makespan_s"]
    return out

# Simulated setup -> the label under which the real run was recorded.
PAIRS = [("S1-QPS01",  "case2_thr_inf"),     ("S2-QPS01",  "case1_thr"),
         ("S3-QPS01",  "case2_32B_thr"),     ("S4-QPS01",  "case3_thr"),
         ("S1-QPS004", "miro_case2_qps004"), ("S1-QPS001", "miro_case2_qps001")]


def main():
    real = real_runs()
    print(f"{'setup':12s} {'measured':>13s} {'sim wall':>10s} {'speedup':>9s}")
    tot_r = tot_s = 0.0
    sims = []
    for case, label in PAIRS:
        t = os.path.join(ROOT, "simulator_output", "validate", f"{case}.timing.json")
        if not os.path.exists(t) or label not in real:
            print(f"{case:12s} {'(missing)':>13s}")
            continue
        r, sw = real[label], json.load(open(t))["sim_wall_s"]
        tot_r += r; tot_s += sw; sims.append(sw)
        print(f"{case:12s} {r:12.1f}s {sw:9.1f}s {r/sw:8.1f}x")
    if sims:
        print(f"\n{'serial':12s} {tot_r:12.1f}s {tot_s:9.1f}s {tot_r/tot_s:8.1f}x")
        print(f"{'concurrent':12s} {tot_r:12.1f}s {max(sims):9.1f}s {tot_r/max(sims):8.1f}x")


if __name__ == "__main__":
    main()
