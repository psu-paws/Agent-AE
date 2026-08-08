# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Block URLs that mirror public benchmark dumps (GAIA / WebVoyager jsonl, etc.).

Used by web search result filtering and scrape tools so evaluation runs cannot
trivially retrieve answer sheets from the open web. Disable with env:

    BENCHMARK_URL_GUARD=0
"""

from __future__ import annotations

import os


def is_benchmark_url_guard_enabled() -> bool:
    v = os.getenv("BENCHMARK_URL_GUARD", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def is_benchmark_leak_url(url: str) -> bool:
    """
    Return True if this URL should be dropped from search results or blocked for scraping.

    Matching is intentionally conservative on host/path pairs that correspond to known
    leaked datasets; long Etsy ``/market/`` slugs are treated as SEO spam that embeds
    full benchmark questions.
    """
    if not url or not is_benchmark_url_guard_enabled():
        return False

    u = url.strip().lower()
    if u.startswith("http://"):
        u = u[7:]
    elif u.startswith("https://"):
        u = u[8:]

    # WebVoyager / MinorJerry public GAIA mirrors (GitHub UI + raw + forks)
    if "minorjerry/webvoyager" in u:
        return True
    if "gitextract.com" in u and "webvoyager" in u:
        return True

    # Any host: path contains the public dump filename
    if "gaia_web.jsonl" in u:
        return True

    # Raw GitHub JSONL that is part of known agent-benchmark repos
    if "raw.githubusercontent.com" in u and ".jsonl" in u:
        if "webvoyager" in u or "minorjerry" in u or "gaia_web" in u:
            return True

    # Known third-party PDF that duplicated GAIA task text in search indexes
    if "rivista.ai/wp-content/uploads/2025/06/2505.23885v1.pdf" in u:
        return True

    # Etsy SEO listings: entire GAIA question as URL slug
    if "etsy.com" in u and "/market/" in u:
        slug = u.split("/market/", 1)[-1]
        slug = slug.split("?", 1)[0].split("#", 1)[0]
        if len(slug) > 180:
            return True

    return False


BLOCKED_SCRAPE_MESSAGE = (
    "This URL is blocked because it mirrors a public benchmark answer sheet or "
    "dataset dump. Use primary sources (official sites, papers, APIs) instead."
)
