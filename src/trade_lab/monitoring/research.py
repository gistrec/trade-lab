"""Read-only loader for the project's research writeups.

Powers the dashboard's **Research** tab and **About** modal: lists and reads
the markdown corpus (findings, strategy docs, results analyses, methodology,
and the RESULTS master index) straight from the repo. No writes, no exchange,
no credentials — the same read-only contract as the rest of ``monitoring/``.
Only allow-listed paths are read (no path traversal), and a missing file
degrades to a friendly note rather than raising.
"""
from __future__ import annotations

import re
from pathlib import Path

# app.py is src/trade_lab/monitoring/app.py; parents[3] is the repo root, where
# findings/, docs/ and RESULTS.md live (git-tracked, pulled on deploy).
_REPO = Path(__file__).resolve().parents[3]

RESULTS_INDEX = "RESULTS.md"

# Ordered, curated groups (label -> repo-relative markdown paths), in the order
# a reviewer would want to read: deployable strategy first, then the rest.
GROUPS: dict[str, list[str]] = {
    "Deployable strategy": [
        "findings/han_28d_tsmom.md",
        "findings/market_basket_tsmom.md",
        "findings/cluster_stability.md",
        "findings/production_config_v1.md",
    ],
    "Overlays & wrappers": [
        "findings/vol_targeting_regime_gate.md",
        "findings/breadth_filter.md",
        "findings/ensemble_portfolio.md",
    ],
    "Rejected / inconclusive": [
        "findings/cross_sectional_reversal.md",
        "findings/ctrend_proxy_price_only.md",
        "findings/mvrv_overlay.md",
        "findings/hmm_regime_overlay.md",
        "findings/buy_and_hold_cost_symmetry.md",
    ],
    "Validation": [
        "findings/validation_multiexchange.md",
        "findings/validation_execution.md",
        "findings/validation_behavioral_fingerprint.md",
        "findings/validation_lookahead_audit.md",
        "findings/validation_universe_bias.md",
    ],
    "Literature & sessions": [
        "findings/literature_review_v1.md",
        "findings/literature_review_v2.md",
        "findings/literature_review_v3.md",
        "findings/strategy_test_session_2026_05_29.md",
    ],
    "Strategy reference": [
        "docs/strategies/tsmom.md",
        "docs/strategies/sma_cross.md",
        "docs/strategies/regime_sma_cross.md",
        "docs/strategies/regime_only.md",
        "docs/strategies/donchian_trend.md",
        "docs/strategies/pma_ratio.md",
        "docs/strategies/rsi.md",
        "docs/strategies/cross_sectional_momentum.md",
    ],
    "Results analyses": [
        "docs/results/strategy_comparison.md",
        "docs/results/walk_forward_btc.md",
        "docs/results/walk_forward_priority5.md",
        "docs/results/dsr_in_walk_forward.md",
        "docs/results/vol_targeting.md",
        "docs/results/multi_asset.md",
        "docs/results/pit_universe.md",
        "docs/results/yearly_btc.md",
    ],
    "Methodology": [
        "docs/article_validation_methodology_draft.md",
        "docs/backtest_validation.md",
    ],
}


def all_docs() -> list[str]:
    """Every research doc path, flat, RESULTS.md first."""
    return [RESULTS_INDEX, *[p for paths in GROUPS.values() for p in paths]]


_ALLOWED = frozenset(all_docs())


def read_markdown(relpath: str) -> str:
    """Markdown of an allow-listed research doc, or a friendly note if absent.

    Only paths in the curated allowlist are read — never an arbitrary path.
    A missing/unreadable file degrades to a note, never raises.
    """
    if relpath not in _ALLOWED:
        return f"_Unknown document: `{relpath}`._"
    try:
        return (_REPO / relpath).read_text(encoding="utf-8")
    except OSError as exc:
        return f"_Could not read `{relpath}` ({type(exc).__name__})._"


def doc_title(relpath: str) -> str:
    """First ``# `` heading of a doc (its title); falls back to the filename."""
    for line in read_markdown(relpath).splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return Path(relpath).stem.replace("_", " ")


def results_markdown() -> str:
    """The RESULTS.md master-index markdown."""
    return read_markdown(RESULTS_INDEX)


# Backtick-wrapped repo-relative doc paths, e.g. `findings/foo.md` or
# `RESULTS.md`. The char class excludes backticks/spaces so a match can't span
# code fences or run past the closing backtick.
_DOC_PATH_RE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.md)`")


def with_github_links(md: str, repo_url: str, ref: str = "main") -> str:
    """Turn backtick doc-path references into links to the file on GitHub.

    ```findings/foo.md``` becomes ``[`findings/foo.md`](repo/blob/main/findings/foo.md)`` —
    the text stays inline code (monospace, unchanged look), only now it's a
    link. Only **allow-listed** research docs are linked, so a private /
    gitignored / nonexistent path (e.g. ``systems_visibility_roadmap.md``)
    stays plain code rather than becoming a dead link. A no-op when
    ``repo_url`` is empty (links are opt-in, same as the commit links).
    """
    if not repo_url:
        return md
    base = repo_url.rstrip("/")

    def repl(m: "re.Match[str]") -> str:
        path = m.group(1)
        if path not in _ALLOWED:
            return m.group(0)  # unknown/private path — leave as plain code
        return f"[`{path}`]({base}/blob/{ref}/{path})"

    return _DOC_PATH_RE.sub(repl, md)
