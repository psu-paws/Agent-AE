# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Fill Hydra cfg from process environment after .env is loaded (see settings.load_dotenv)."""

from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf

_API_KEY_PLACEHOLDERS = frozenset(
    {"", "EMPTY", "empty", "null", "xxx", "XXX", "???", "your-api-key", "YOUR_API_KEY"}
)


def apply_env_to_hydra_cfg(cfg: DictConfig) -> None:
    """
    If llm.api_key / llm.base_url are unset or placeholder, use env:

    - LLM_API_KEY, then OPENAI_API_KEY, then ANTHROPIC_API_KEY (first non-empty)
    - LLM_BASE_URL when base_url is empty or contains the default placeholder host
    - LLM_MAX_CONTEXT_LENGTH / LLM_MAX_TOKENS when set (must match vLLM --max-model-len budget)
    """
    if not OmegaConf.is_config(cfg) or "llm" not in cfg:
        return

    OmegaConf.set_struct(cfg, False)
    try:
        api_key = cfg.llm.get("api_key")
        api_key_s = (str(api_key).strip() if api_key is not None else "") or ""
        if api_key_s in _API_KEY_PLACEHOLDERS:
            for env_key in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                v = os.environ.get(env_key, "").strip()
                if v:
                    cfg.llm.api_key = v
                    break

        base_url = cfg.llm.get("base_url")
        base_s = (str(base_url).strip() if base_url is not None else "") or ""
        env_base = os.environ.get("LLM_BASE_URL", "").strip()
        if env_base and (not base_s or "your-api.com" in base_s):
            cfg.llm.base_url = env_base

        mctx = os.environ.get("LLM_MAX_CONTEXT_LENGTH", "").strip()
        if mctx:
            try:
                cfg.llm.max_context_length = int(mctx)
            except ValueError:
                pass
        mtok = os.environ.get("LLM_MAX_TOKENS", "").strip()
        if mtok:
            try:
                cfg.llm.max_tokens = int(mtok)
            except ValueError:
                pass
    finally:
        OmegaConf.set_struct(cfg, True)
