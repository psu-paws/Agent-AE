# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Per-run output directory layout (traces, benchmark results, summarizer I/O)."""

from __future__ import annotations

import os
from pathlib import Path


def task_traces_dir(run_root: str | Path) -> Path:
    """Directory where task_*.json trace files are written for this run."""
    p = Path(run_root).expanduser().resolve() / "traces"
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_task_trace_json_files(run_root: str | Path) -> list[Path]:
    """Task trace JSON paths: prefer traces/, fall back to legacy flat layout."""
    root = Path(run_root).expanduser().resolve()
    td = root / "traces"
    if td.is_dir():
        found = sorted(td.glob("task_*.json"))
        if found:
            return found
    return sorted(root.glob("task_*.json"))


def configure_run_artifact_env(log_dir: str | Path) -> Path:
    """
    Ensure log_dir exists, create summarizer_io/, and point SUMMARIZER_LLM_IO_LOG_DIR there
    when the user has not set that variable (e.g. in .env).
    """
    p = Path(log_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    task_traces_dir(p)
    summarizer_dir = p / "summarizer_io"
    summarizer_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("SUMMARIZER_LLM_IO_LOG_DIR", "").strip():
        os.environ["SUMMARIZER_LLM_IO_LOG_DIR"] = str(summarizer_dir)
    return p
