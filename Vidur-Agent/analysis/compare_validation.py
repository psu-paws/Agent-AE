"""Compare simulated session latency against the measured deployment.

Both sides are reduced with the same formulas here rather than reusing each
side's own summary script, because those define session latency differently and
the gap would be misread as simulator error.

    trace time   = max(completion) - min(arrival)   over all requests
    session e2e  = max(completion) - min(arrival)   within one session

The measured side uses the client's wall clock, which is what a user sees. The
simulator's Request Id is the trace's request_id verbatim, so the join is exact.
"""
import json, os, sys
import numpy as np
import pandas as pd

GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ground_truth")
CASES = [
    ("S1", "S1-QPS01",  f"{GT}/miro_S1_0.1"),
    ("S2", "S2-QPS01",  f"{GT}/miro_S2_0.1"),
    ("S3", "S3-QPS01",  f"{GT}/miro_S3_0.1"),
    ("S4", "S4-QPS01",  f"{GT}/miro_S4_0.1"),
    ("S1 @0.04", "S1-QPS004", f"{GT}/miro_S1_0.04"),
    ("S1 @0.01", "S1-QPS001", f"{GT}/miro_S1_0.01"),
]

def geomean(x):
    x = np.asarray(x, float); x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")

def stats(spans):
    s = pd.Series(spans, dtype=float)
    return {"n": int(len(s)), "mean": float(s.mean()), "geomean": geomean(s),
            "p50": float(s.quantile(.5)), "p90": float(s.quantile(.9)),
            "p99": float(s.quantile(.99)), "max": float(s.max())}

# request -> session comes from the ground truth itself; the simulator indexes
# requests in trace order, so the mapping is the same on both sides.
sess_map = (pd.read_csv(f"{CASES[0][2]}/raw_results.csv", usecols=["request_id", "session_id"])
              .drop_duplicates("request_id"))

def reduce_gt(run):
    d = pd.read_csv(os.path.join(run, "raw_results.csv"))
    if "success" in d.columns: d = d[d["success"] == True]
    d = d.rename(columns={"client_sent_wall": "arr", "client_recv_wall": "comp"})
    per = d.groupby("session_id").agg(a=("arr", "min"), c=("comp", "max"))
    return float(d["comp"].max() - d["arr"].min()), stats(per["c"] - per["a"]), len(d)

def reduce_sim(run):
    m = pd.read_csv(os.path.join(run, "request_metrics.csv"))
    m["comp"] = m["request_arrived_at"] + m["request_e2e_time"]
    per_req = m.groupby("Request Id").agg(arr=("request_arrived_at", "min"), comp=("comp", "max")).reset_index()
    per_req = per_req.merge(sess_map, left_on="Request Id", right_on="request_id", how="left")
    if per_req["session_id"].isna().any():
        print("  WARN: %d requests did not join to a session" % per_req["session_id"].isna().sum())
    per = per_req.groupby("session_id").agg(a=("arr", "min"), c=("comp", "max"))
    return float(per_req["comp"].max() - per_req["arr"].min()), stats(per["c"] - per["a"]), len(per_req)

rows = []
for label, simdir, gtdir in CASES:
    sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "simulator_output", "validate", simdir)
    if not os.path.exists(os.path.join(sp, "request_metrics.csv")):
        print(f"skip {label}: no sim output"); continue
    if not os.path.exists(os.path.join(gtdir, "raw_results.csv")):
        print(f"skip {label}: no ground truth at {gtdir}"); continue
    g_mk, g_s, g_n = reduce_gt(gtdir)
    s_mk, s_s, s_n = reduce_sim(sp)
    rows.append((label, g_mk, s_mk, g_s, s_s, g_n, s_n))

def err(sim, gt): return 100.0 * (sim - gt) / gt if gt else float("nan")

print("\n" + "=" * 92)
print("MAKESPAN")
print("=" * 92)
print(f"{'case':18s} {'measured':>12s} {'simulated':>12s} {'error':>9s}   {'reqs gt/sim':>14s}")
for label, g, s, _, _, gn, sn in rows:
    print(f"{label:18s} {g:11.1f}s {s:11.1f}s {err(s,g):+8.1f}%   {gn:6d}/{sn:<7d}")

for key in ("geomean", "mean", "p50", "p90", "p99"):
    print("\n" + "=" * 92)
    print(f"SESSION E2E  --  {key}")
    print("=" * 92)
    print(f"{'case':18s} {'measured':>12s} {'simulated':>12s} {'error':>9s}   {'sessions gt/sim':>16s}")
    for label, _, _, gs, ss, _, _ in rows:
        print(f"{label:18s} {gs[key]:11.1f}s {ss[key]:11.1f}s {err(ss[key],gs[key]):+8.1f}%   {gs['n']:7d}/{ss['n']:<8d}")
print()
