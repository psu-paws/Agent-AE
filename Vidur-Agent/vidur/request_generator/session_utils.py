"""Session and turn identity for trace-driven request generation.

A task ("session") is a sequence of requests ("turns") where later turns often
reuse the KV cache of earlier ones. Session identity drives session-priority
scheduling, inter-turn arrival chaining and dependency release, so it has to be
derived unambiguously.

Traces identify their turns in one of three ways, tried in this order:

1. ``session_id`` + ``turn_id`` — explicit and preferred. Write traces this way.
2. ``session_id`` alone — turn index is the row's position within its session.
3. Neither — each request becomes its own single-turn session.

``request_id``, where present, is treated as an opaque unique identifier only;
nothing is decoded from its value.
"""

import pandas as pd

from vidur.logger import init_logger

logger = init_logger(__name__)

def _has_column(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns and df[name].notna().all()


def _densify_session_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure session ids are exactly 0..N-1.

    Downstream code indexes per-session state by position (``session_row_indices``,
    ``num_requests_issued``), so sparse or non-zero-based session ids would go out
    of range. Traces that are already dense are left untouched, which keeps the
    ids stable for the traces shipped with this repo.
    """
    unique = df["session_id"].unique()
    if set(unique) == set(range(len(unique))):
        return df

    logger.warning(
        f"session_id values are not dense 0..{len(unique) - 1}; renumbering "
        "by order of first appearance"
    )
    df["session_id"] = pd.factorize(df["session_id"])[0]
    return df


def resolve_session_and_turn(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized ``session_id`` and ``turn_id`` columns to ``df``.

    Both are plain 0-based integers. ``df`` is modified in place and also
    returned. Row order is preserved; turn indices in case 2 follow it.
    """
    has_session = _has_column(df, "session_id")
    has_turn = _has_column(df, "turn_id")

    if has_session and has_turn:
        df["session_id"] = df["session_id"].astype(int)
        df["turn_id"] = df["turn_id"].astype(int)
        logger.info("Session identity: explicit session_id + turn_id columns")
        return _densify_session_ids(df)

    if has_session:
        df["session_id"] = df["session_id"].astype(int)
        df["turn_id"] = df.groupby("session_id").cumcount()
        logger.info("Session identity: session_id column, turn_id from row order")
        return _densify_session_ids(df)

    # Flat trace: independent requests, one single-turn session each.
    df["session_id"] = range(len(df))
    df["turn_id"] = 0
    logger.info(
        f"Session identity: no session_id column, treating all {len(df)} requests "
        "as independent single-turn sessions"
    )
    return df
