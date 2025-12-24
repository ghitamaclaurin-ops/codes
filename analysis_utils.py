# -*- coding: utf-8 -*-
"""
Shared utilities for publication-ready statistical figures and documentation.

Each analysis script imports this module to ensure that:
- output files are written into the local ``figures_results`` directory
- matplotlib uses a consistent grayscale style appropriate for journals
- significance annotations and Tukey parsing behave identically
- global Markdown documents (figure legends & results summary) stay in sync
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths & Matplotlib style
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTDIR = PROJECT_ROOT / "figures_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

FONT_FAMILY = ["DejaVu Sans", "Arial", "Liberation Sans"]
BAR_WIDTH = 0.55
PANEL_WIDTH = 4.0
PANEL_HEIGHT = 4.0


def apply_publication_style() -> None:
    """Apply a consistent grayscale-friendly matplotlib style."""
    matplotlib.rcParams["font.family"] = FONT_FAMILY[0]
    matplotlib.rcParams["font.sans-serif"] = FONT_FAMILY
    matplotlib.rcParams["axes.labelsize"] = 10
    matplotlib.rcParams["axes.titlesize"] = 11
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["axes.edgecolor"] = "black"
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["figure.dpi"] = 150
    matplotlib.rcParams["savefig.dpi"] = 300


apply_publication_style()

GROUP_ORDER = ["NC", "MC", "LD", "HD"]

# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------


def p_to_stars(p_value: float) -> str:
    """Convert a p-value to the journal-friendly star notation."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def tukey_pairs_with_stars(table, group1_col: str = "group1", group2_col: str = "group2",
                           pvalue_col: str | None = None) -> List[Tuple[str, str, float, str]]:
    """
    Convert a Tukey HSD summary table/DataFrame into ``(g1, g2, p, stars)`` tuples.
    Supports both ``p-adj`` and ``p_adj`` column names.
    """
    if hasattr(table, "to_dict"):
        records = list(table.to_dict("records"))
    else:
        records = list(table)

    if not records:
        return []

    if pvalue_col is None:
        for candidate in ("p-adj", "p_adj", "p_adj."):
            if candidate in records[0]:
                pvalue_col = candidate
                break
        if pvalue_col is None:
            raise KeyError("Unable to locate adjusted p-value column in Tukey table")

    pairs: List[Tuple[str, str, float, str]] = []
    for record in records:
        g1 = record[group1_col]
        g2 = record[group2_col]
        p_val = float(record[pvalue_col])
        pairs.append((g1, g2, p_val, p_to_stars(p_val)))
    return pairs


def add_significance_annotations(ax, groups: Sequence[str], means, sds, pairs,
                                 control_group: str | None = "MC",
                                 only_significant: bool = True) -> None:
    """
    Draw Tukey-style brackets with significance stars on a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    groups : sequence
        Ordered group labels corresponding to ``means`` and ``sds``.
    means, sds : array-like
        Mean and standard deviation values aligned with ``groups``.
    pairs : iterable of tuples
        ``(group1, group2, p, stars)`` entries.
    control_group : str or None
        If provided, only comparisons versus this group will be annotated.
        Set to ``None`` to annotate all pairwise combinations.
    only_significant : bool
        Skip ``ns`` comparisons when True.
    """
    groups = list(groups)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    if means.size == 0:
        return

    top = np.nanmax(means + np.nan_to_num(sds))
    height = top * 0.05 if top > 0 else 0.1
    step = top * 0.08 if top > 0 else 0.15
    current_y = top + step

    for g1, g2, p_val, stars in pairs:
        if only_significant and stars == "ns":
            continue
        if control_group and control_group not in (g1, g2):
            continue
        if g1 not in groups or g2 not in groups:
            continue
        i, j = groups.index(g1), groups.index(g2)
        x1, x2 = sorted((i, j))
        ax.plot([x1, x1, x2, x2],
                [current_y, current_y + height, current_y + height, current_y],
                c="black", lw=1.2)
        ax.text((x1 + x2) / 2, current_y + height, stars,
                ha="center", va="bottom", fontsize=10)
        current_y += step


def get_pair_result(pairs: Iterable[Tuple[str, str, float, str]],
                    group_a: str, group_b: str) -> Tuple[float | None, str]:
    """
    Retrieve the ``(p_value, stars)`` tuple for a specific comparison from ``pairs``.
    Returns ``(None, "ns")`` if the combination is absent.
    """
    for g1, g2, p_val, stars in pairs:
        if {g1, g2} == {group_a, group_b}:
            return p_val, stars
    return None, "ns"


def format_p_value(p_value: float | None) -> str:
    """Return a compact p-value string for figure legends / results."""
    if p_value is None or np.isnan(p_value):
        return "p>0.05"
    if p_value < 0.001:
        return "p<0.001"
    if p_value < 0.01:
        return "p<0.01"
    if p_value < 0.05:
        return "p<0.05"
    return f"p={p_value:.3f}"


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def render_bar_panel(ax,
                     panel_label: str,
                     panel_title: str,
                     *,
                     groups: Sequence[str] | None = None,
                     means: Sequence[float] | np.ndarray,
                     sds: Sequence[float] | np.ndarray,
                     tukey_pairs: Iterable[Tuple[str, str, float, str]] | None = None,
                     control_group: str | None = "MC",
                     only_significant: bool = True,
                     ylabel: str | None = None,
                     bar_color: str = "grey",
                     edge_color: str = "black"):
    """
    Draw a publication-ready bar panel with Tukey annotations.

    Parameters mirror the pattern used across all analysis scripts so that
    stylistic updates remain centralized.
    """
    if groups is None:
        groups = GROUP_ORDER
    groups = list(groups)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    if means.size != len(groups) or sds.size != len(groups):
        raise ValueError("`means` and `sds` must align with `groups` length")

    x = np.arange(len(groups))
    ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=bar_color,
        edgecolor=edge_color,
        width=BAR_WIDTH,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(ylabel or panel_title, fontsize=10)
    ax.set_title(f"{panel_label}. {panel_title}", fontsize=11, fontweight="bold")

    if tukey_pairs:
        add_significance_annotations(
            ax,
            groups,
            means,
            sds,
            tukey_pairs,
            control_group=control_group,
            only_significant=only_significant,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return ax


def panel_figsize(nrows: int, ncols: int,
                  width_per_col: float | None = None,
                  height_per_row: float | None = None) -> tuple[float, float]:
    """
    Compute a consistent ``figsize`` for multi-panel figures so bar widths match
    regardless of grid layout.
    """
    width = width_per_col or PANEL_WIDTH
    height = height_per_row or PANEL_HEIGHT
    return (ncols * width, nrows * height)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

FIGURE_LEGENDS_FILE = OUTDIR / "figure_legends.md"
RESULTS_SUMMARY_FILE = OUTDIR / "results_summary.md"


def export_figure(fig, stem_name: str, outdir: Path = OUTDIR,
                  formats: Sequence[str] = ("svg", "pdf")) -> dict:
    """Save ``fig`` to the desired formats inside ``outdir``."""
    out_paths = {}
    for ext in formats:
        path = outdir / f"{stem_name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight")
        out_paths[ext] = path
    return out_paths


def write_text_file(filename: str, content: str, outdir: Path = OUTDIR) -> Path:
    """Write ``content`` to ``outdir / filename`` (UTF-8) and return the path."""
    target = outdir / filename
    target.write_text(content.strip() + "\n", encoding="utf-8")
    return target


def _upsert_markdown_section(file_path: Path, section_title: str,
                             body: str, top_heading: str) -> None:
    """Insert or replace a markdown section with ``## section_title``."""
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
    else:
        text = f"# {top_heading}\n\n"

    section_header = f"## {section_title}\n"
    section_block = section_header + body.strip() + "\n"
    pattern = re.compile(rf"(## {re.escape(section_title)}\n)(.*?)(?=\n## |\Z)", re.S)

    if pattern.search(text):
        text = pattern.sub(section_block + "\n", text)
    else:
        if not text.endswith("\n\n"):
            text = text.rstrip() + "\n\n"
        text += section_block + "\n"

    file_path.write_text(text, encoding="utf-8")


def record_figure_legend(figure_id: str, caption: str,
                         panels: Sequence[Tuple[str, str]] | None = None,
                         stats_note: str | None = None) -> None:
    """Add or update a figure legend entry inside ``figure_legends.md``."""
    lines = [f"**Caption:** {caption}"]
    if panels:
        lines.append("**Panels:**")
        for label, desc in panels:
            lines.append(f"- {label}: {desc}")
    if stats_note:
        lines.append(f"**Statistics:** {stats_note}")
    body = "\n".join(lines)
    _upsert_markdown_section(FIGURE_LEGENDS_FILE, figure_id, body,
                             top_heading="Figure Legends")


def record_results_section(section_title: str,
                           highlights: Sequence[str],
                           methods_note: str | None = None) -> None:
    """Add a concise Result-style paragraph for the section."""
    lines = ["**Key findings:**"]
    for item in highlights:
        lines.append(f"- {item}")
    if methods_note:
        lines.append(f"**Methods:** {methods_note}")
    body = "\n".join(lines)
    _upsert_markdown_section(RESULTS_SUMMARY_FILE, section_title, body,
                             top_heading="Results Summary")
