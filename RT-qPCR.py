# -*- coding: utf-8 -*-
"""
RT-qPCR inflammatory gene expression analysis.

Outputs CSV tables, ANOVA/Tukey statistics, and a 2x3 multi-panel figure (Fig5).
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from textwrap import dedent

from analysis_utils import (
    OUTDIR,
    GROUP_ORDER,
    export_figure,
    format_p_value,
    get_pair_result,
    panel_figsize,
    record_figure_legend,
    record_results_section,
    render_bar_panel,
    tukey_pairs_with_stars,
    write_text_file,
)

MARKERS = ["caspase1", "IL1b", "MCP1", "IL18", "NLRP3", "VCAM1"]
PANEL_LABELS = ["G", "H", "I", "J", "K", "L"]
DISPLAY_LABELS = {
    "caspase1": "Caspase-1 (fold change)",
    "IL1b": "IL-1β (fold change)",
    "MCP1": "MCP-1 (fold change)",
    "IL18": "IL-18 (fold change)",
    "NLRP3": "NLRP3 (fold change)",
    "VCAM1": "VCAM-1 (fold change)",
}
PANEL_TITLES = {
    "caspase1": "Caspase-1 expression",
    "IL1b": "IL-1β expression",
    "MCP1": "MCP-1 expression",
    "IL18": "IL-18 expression",
    "NLRP3": "NLRP3 expression",
    "VCAM1": "VCAM-1 expression",
}

# Data loaded from CSV file
# RAW_DATA dictionary replaced by CSV import


def create_dataframe() -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv("data/qpcr_data.csv")


def summarize(df: pd.DataFrame):
    summary = (
        df.groupby("Group")[MARKERS]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary.reset_index(inplace=True)
    return summary


def run_statistics(df: pd.DataFrame):
    stats = {}
    for marker in MARKERS:
        model = ols(f"{marker} ~ C(Group)", data=df).fit()
        anova_tab = anova_lm(model, typ=2)
        tukey = pairwise_tukeyhsd(df[marker], df["Group"], alpha=0.05)
        header = tukey.summary().data[0]
        rows = tukey.summary().data[1:]
        tukey_df = pd.DataFrame(rows, columns=header)
        stats[marker] = {
            "anova": anova_tab,
            "tukey": tukey_df,
            "pairs": tukey_pairs_with_stars(tukey_df),
        }
    return stats


def save_tables(df: pd.DataFrame, summary: pd.DataFrame, stats: dict):
    raw_csv = OUTDIR / "rt_qpcr_raw.csv"
    sum_csv = OUTDIR / "rt_qpcr_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "rt_qpcr_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for marker, result in stats.items():
            result["anova"].to_excel(writer, sheet_name=f"{marker}_ANOVA")
            result["tukey"].to_excel(writer, sheet_name=f"{marker}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_rt_qpcr(summary: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(2, 3, figsize=panel_figsize(2, 3))
    axes = axes.flatten()

    for idx, marker in enumerate(MARKERS):
        ax = axes[idx]
        means = [summary.loc[summary["Group"] == g, f"{marker}_mean"].values[0] for g in GROUP_ORDER]
        sds = [summary.loc[summary["Group"] == g, f"{marker}_std"].values[0] for g in GROUP_ORDER]
        render_bar_panel(
            ax,
            panel_label=PANEL_LABELS[idx],
            panel_title=PANEL_TITLES[marker],
            groups=GROUP_ORDER,
            means=means,
            sds=sds,
            tukey_pairs=stats[marker]["pairs"],
            ylabel="Relative expression",
        )

    plt.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def write_readme(raw_csv, sum_csv, xlsx_path, figure_paths):
    content = dedent(f"""
    RT-qPCR炎症基因结果（Fig.5G–L）
    ===========================

    文件说明:
      - rt_qpcr_raw.csv / summary.csv / stats.xlsx
      - Fig5_RT_qPCR_G-L.svg / pdf

    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_rt_qpcr.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    highlights = []
    for marker in MARKERS:
        mc = summary.loc[summary["Group"] == "MC", f"{marker}_mean"].values[0]
        hd = summary.loc[summary["Group"] == "HD", f"{marker}_mean"].values[0]
        ld = summary.loc[summary["Group"] == "LD", f"{marker}_mean"].values[0]
        p_ld, _ = get_pair_result(stats[marker]["pairs"], "MC", "LD")
        p_hd, _ = get_pair_result(stats[marker]["pairs"], "MC", "HD")
        highlights.append(
            f"{marker} expression was MC {mc:.2f}, reduced to LD {ld:.2f} ({format_p_value(p_ld)}) "
            f"and HD {hd:.2f} ({format_p_value(p_hd)})."
        )

    record_results_section(
        "Figure 5 (G–L) - RT-qPCR",
        highlights,
        methods_note="Mean ± SD, one-way ANOVA with Tukey HSD vs MC.",
    )

    record_figure_legend(
        "Fig5_RT_qPCR_G-L",
        "RT-qPCR validation of inflammasome and adhesion genes (panels G–L).",
        panels=[(label, DISPLAY_LABELS[marker]) for label, marker in zip(PANEL_LABELS, MARKERS)],
        stats_note="Bars denote mean ± SD. Tukey HSD annotations vs MC.",
    )


def main():
    df = create_dataframe()
    summary = summarize(df)
    stats = run_statistics(df)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_rt_qpcr(summary, stats)
    figure_paths = export_figure(fig, "Fig5_RT_qPCR_G-L")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
