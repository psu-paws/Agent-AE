# Ground-truth measurements

Per-session and per-request results from the real Dynamo + vLLM deployment the
simulator is validated against, one directory per setup (`<trace>_S<n>_<rate>`).
Reproducing these needs GPUs and a Dynamo install, so the measurements are
included here directly and `analysis/compare_validation.py` reads them from this path.

- `raw_results.csv` — one row per request: token counts, and the client-side
  send/receive wall clock the session latencies are computed from.
- `run_summary.json` — request and session counts, trace time, and the latency
  percentiles in the paper.
