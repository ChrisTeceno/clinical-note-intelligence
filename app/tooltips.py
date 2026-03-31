"""Shared CSS tooltip utilities for the Streamlit dashboard.

Streamlit's `unsafe_allow_html` strips `title` and `data-*` attributes,
so we use a nested-span approach with pure CSS visibility toggling.
"""

import math

# CSS injected once per page via st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)
TOOLTIP_CSS = """<style>
.tt { position: relative; display: inline-block; cursor: help; }
.tt .ttt {
    visibility: hidden; opacity: 0;
    background: #333; color: #fff;
    padding: 6px 10px; border-radius: 6px;
    font-size: 0.78em; font-weight: normal;
    position: absolute; z-index: 1000;
    bottom: 125%; left: 50%; transform: translateX(-50%);
    width: max-content; max-width: 280px;
    line-height: 1.4; text-align: left;
    transition: opacity 0.15s;
    pointer-events: none;
}
.tt:hover .ttt { visibility: visible; opacity: 1; }
</style>"""


def tt(label_html: str, tip: str) -> str:
    """Wrap *label_html* in a tooltip span with *tip* shown on hover.

    Parameters
    ----------
    label_html : str
        Inner HTML for the visible element (may contain inline styles).
    tip : str
        Plain-text explanation shown on hover.

    Returns
    -------
    str
        HTML snippet safe for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    # Escape only the tooltip text to avoid breaking nested HTML
    safe_tip = (
        tip.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&#39;")
    )
    return f'<span class="tt">{label_html}<span class="ttt">{safe_tip}</span></span>'


def tt_heading(text: str, tip: str, tag: str = "span", style: str = "") -> str:
    """Return a heading-like element with a small info icon that has a tooltip.

    Useful for chart/section titles that need hover explanations.
    """
    icon_html = (
        '<span style="font-size:0.7em;color:#888;'
        'vertical-align:super;margin-left:4px">&#9432;</span>'
    )
    info_icon = tt(icon_html, tip)
    if style:
        return f"<{tag} style='{style}'>{text}{info_icon}</{tag}>"
    return f"<{tag}>{text}{info_icon}</{tag}>"


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a proportion.

    Returns (lower, upper) as fractions in [0, 1].
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_pct_ci(successes: int, total: int) -> str:
    """Format a percentage with its 95% Wilson CI, e.g. '86.2% [95% CI: 80.4% - 90.6%]'."""
    if total == 0:
        return "N/A"
    pct = successes / total * 100
    lo, hi = wilson_ci(successes, total)
    return f"{pct:.1f}% [95% CI: {lo*100:.1f}% \u2013 {hi*100:.1f}%]"
