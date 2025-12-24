# -*- coding: utf-8 -*-
"""
Immunohistochemistry (IHC) semi-quantitative scoring for inflammasome markers.

The script parses the provided mean ± SD table, performs ANOVA/Tukey statistics,
and generates a figure consistent with the rest of the pipeline.
"""

from __future__ import annotations

import re
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

GENES = ["caspase1", "IL1b", "MCP1", "IL18", "NLRP3", "VCAM1"]
DISPLAY_LABELS = {
    "caspase1": "Caspase-1 (IHC score)",
    "IL1b": "IL-1β (IHC score)",
    "MCP1": "MCP-1 (IHC score)",
    "IL18": "IL-18 (IHC score)",
    "NLRP3": "NLRP3 (IHC score)",
    "VCAM1": "VCAM-1 (IHC score)",
}
PANEL_TITLES = {
    "caspase1": "Caspase-1 staining",
    "IL1b": "IL-1β staining",
    "MCP1": "MCP-1 staining",
    "IL18": "IL-18 staining",
    "NLRP3": "NLRP3 staining",
    "VCAM1": "VCAM-1 staining",
}
PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]

# RAW_ROWS data replaced by CSV import

# GROUP_MAP replaced by CSV Group column


def parse_mean_sd(entry: str):
    if "±" in entry:
        mean, sd = entry.split("±")
        return float(mean), float(sd)
    parts = re.split(r"±|\+/-", entry)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    return float(entry), 0.0


def create_dataframe() -> pd.DataFrame:
    """Load data from CSV and compute mean values per sample."""
    df_raw = pd.read_csv("data/ihc_data.csv")
    # Average across replicates for each sample
    df_avg = df_raw.groupby(['Sample', 'Group'])[GENES].mean().reset_index()

    # Convert to long format for analysis
    records = []
    for _, row in df_avg.iterrows():
        sample = row['Sample']
        group = row['Group']
        for gene in GENES:
            records.append({
                "sample": str(sample),
                "group": group,
                "gene": gene,
                "value": row[gene],
                "input_sd": 0.0,
            })
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame):
    summary = (
        df.groupby(["gene", "group"])
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
        .reset_index()
    )
    summary["mean_sd"] = summary["mean"].round(2).astype(str) + " ± " + summary["sd"].round(2).astype(str)
    return summary


def run_statistics(df: pd.DataFrame):
    stats = {}
    for gene in GENES:
        sub = df[df["gene"] == gene]
        model = ols("value ~ C(group)", data=sub).fit()
        anova_tab = anova_lm(model, typ=2)
        tukey = pairwise_tukeyhsd(sub["value"], sub["group"], alpha=0.05)
        header = tukey.summary().data[0]
        rows = tukey.summary().data[1:]
        tukey_df = pd.DataFrame(rows, columns=header)
        stats[gene] = {
            "anova": anova_tab,
            "tukey": tukey_df,
            "pairs": tukey_pairs_with_stars(tukey_df),
        }
    return stats


def save_tables(df: pd.DataFrame, summary: pd.DataFrame, stats: dict):
    raw_csv = OUTDIR / "ihc_sample_data.csv"
    sum_csv = OUTDIR / "ihc_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "ihc_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for gene, res in stats.items():
            res["anova"].to_excel(writer, sheet_name=f"{gene}_ANOVA")
            res["tukey"].to_excel(writer, sheet_name=f"{gene}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_ihc(summary: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(2, 3, figsize=panel_figsize(2, 3))
    axes = axes.flatten()

    for idx, gene in enumerate(GENES):
        ax = axes[idx]
        means = [summary.loc[(summary["gene"] == gene) & (summary["group"] == g), "mean"].values[0] for g in GROUP_ORDER]
        sds = [summary.loc[(summary["gene"] == gene) & (summary["group"] == g), "sd"].values[0] for g in GROUP_ORDER]
        render_bar_panel(
            ax,
            panel_label=PANEL_LABELS[idx],
            panel_title=PANEL_TITLES[gene],
            groups=GROUP_ORDER,
            means=means,
            sds=sds,
            tukey_pairs=stats[gene]["pairs"],
            ylabel=DISPLAY_LABELS[gene],
        )

    plt.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def write_readme(raw_csv, sum_csv, xlsx_path, figure_paths):
    content = dedent(f"""
    IHC炎症标记结果
    =============

    文件说明:
      - ihc_sample_data.csv / ihc_summary.csv / ihc_stats.xlsx
      - Fig5_IHC.svg / pdf

    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_ihc.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    highlights = []
    for gene in GENES:
        mc = summary.loc[(summary["gene"] == gene) & (summary["group"] == "MC"), "mean"].values[0]
        hd = summary.loc[(summary["gene"] == gene) & (summary["group"] == "HD"), "mean"].values[0]
        ld = summary.loc[(summary["gene"] == gene) & (summary["group"] == "LD"), "mean"].values[0]
        p_ld, _ = get_pair_result(stats[gene]["pairs"], "MC", "LD")
        p_hd, _ = get_pair_result(stats[gene]["pairs"], "MC", "HD")
        highlights.append(
            f"{gene} staining: MC {mc:.2f}, reduced to LD {ld:.2f} ({format_p_value(p_ld)}) "
            f"and HD {hd:.2f} ({format_p_value(p_hd)})."
        )

    record_results_section(
        "Figure 5 - IHC",
        highlights,
        methods_note="Semi-quantitative mean ± SD, one-way ANOVA + Tukey HSD vs MC.",
    )

    record_figure_legend(
        "Fig5_IHC",
        "Immunohistochemistry scoring of inflammasome-related proteins.",
        panels=[
            ("A", "Caspase-1."),
            ("B", "IL-1β."),
            ("C", "MCP-1."),
            ("D", "IL-18."),
            ("E", "NLRP3."),
            ("F", "VCAM-1."),
        ],
        stats_note="Bars show mean ± SD with Tukey HSD vs MC.",
    )


def main():
    df = create_dataframe()
    summary = summarize(df)
    stats = run_statistics(df)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_ihc(summary, stats)
    figure_paths = export_figure(fig, "Fig5_IHC")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
