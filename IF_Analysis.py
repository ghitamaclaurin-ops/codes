# -*- coding: utf-8 -*-
"""
Immunofluorescence (IF) cytokine profiling.

Generates publication-ready statistics and a four-panel grayscale figure for
IL-1β, IL-18, TNF-α, and CRP (Figure 4 in the manuscript).
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

METRICS = ["IL_1B", "IL_18", "TNFA", "CRP"]
UNITS = {"IL_1B": "pg/mL", "IL_18": "pg/mL", "TNFA": "pg/mL", "CRP": "mg/L"}
METRIC_NAMES = {"IL_1B": "IL-1β", "IL_18": "IL-18", "TNFA": "TNF-α", "CRP": "CRP"}
PANEL_LABELS = ["A", "B", "C", "D"]
PANEL_TITLES = {
    "IL_1B": "IL-1β",
    "IL_18": "IL-18",
    "TNFA": "TNF-α",
    "CRP": "CRP",
}
YLABELS = {metric: f"{METRIC_NAMES[metric]} ({UNITS[metric]})" for metric in METRICS}

# Data loaded from CSV file
df = pd.read_csv("data/inflammatory_cytokines_data.csv")
# Rename columns back to original names for compatibility
df = df.rename(columns={"IL1beta": "IL_1B", "IL18": "IL_18", "TNFalpha": "TNFA"})


def create_if_dataframe() -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv("data/inflammatory_cytokines_data.csv")
    # Rename columns back to original names for compatibility
    df = df.rename(columns={"IL1beta": "IL_1B", "IL18": "IL_18", "TNFalpha": "TNFA"})
    return df


def build_long_format(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        for metric in METRICS:
            rows.append({
                "group": row["Group"],
                "metric": metric,
                "value": row[metric],
                "unit": UNITS[metric],
            })
    df_long = pd.DataFrame(rows)
    summary = (
        df_long.groupby(["metric", "group"])
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
        .reset_index()
    )
    summary["mean_sd"] = summary["mean"].round(2).astype(str) + " ± " + summary["sd"].round(2).astype(str)
    return df_long, summary


def run_statistics(df_long: pd.DataFrame):
    stats = {}
    for metric in METRICS:
        sub = df_long[df_long["metric"] == metric]
        model = ols("value ~ C(group)", data=sub).fit()
        anova_tab = anova_lm(model, typ=2)
        tukey = pairwise_tukeyhsd(sub["value"], sub["group"], alpha=0.05)
        header = tukey.summary().data[0]
        rows = tukey.summary().data[1:]
        tukey_df = pd.DataFrame(rows, columns=header)
        stats[metric] = {
            "anova": anova_tab,
            "tukey": tukey_df,
            "pairs": tukey_pairs_with_stars(tukey_df),
        }
    return stats


def save_tables(df: pd.DataFrame, summary: pd.DataFrame, stats: dict):
    raw_csv = OUTDIR / "if_analysis_data.csv"
    sum_csv = OUTDIR / "if_analysis_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "if_analysis_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for metric, res in stats.items():
            res["anova"].to_excel(writer, sheet_name=f"{metric}_ANOVA")
            res["tukey"].to_excel(writer, sheet_name=f"{metric}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_if_panels(df_long: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        sub = df_long[df_long["metric"] == metric]
        means = [sub[sub["group"] == g]["value"].mean() for g in GROUP_ORDER]
        sds = [sub[sub["group"] == g]["value"].std() for g in GROUP_ORDER]
        render_bar_panel(
            ax,
            panel_label=PANEL_LABELS[idx],
            panel_title=PANEL_TITLES[metric],
            groups=GROUP_ORDER,
            means=means,
            sds=sds,
            tukey_pairs=stats[metric]["pairs"],
            ylabel=YLABELS[metric],
        )

    plt.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def write_readme(raw_csv, sum_csv, xlsx_path, figure_paths):
    content = dedent(f"""
    免疫荧光炎症指标
    ==============

    文件说明:
      - if_analysis_data.csv          原始IF荧光强度
      - if_analysis_summary.csv       均值 ± 标准差
      - if_analysis_stats.xlsx        One-way ANOVA + Tukey HSD
      - Fig4_Inflammatory_Markers.*   四面板炎症因子图

    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_if_analysis.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    indexed = summary.set_index(["metric", "group"])

    def describe(metric, group):
        row = indexed.loc[(metric, group)]
        unit = UNITS[metric]
        return f"{row['mean']:.1f} ± {row['sd']:.1f} {unit}"

    highlights = []
    for metric in ("IL_1B", "IL_18", "TNFA", "CRP"):
        p_val, _ = get_pair_result(stats[metric]["pairs"], "MC", "NC")
        highlights.append(
            f"MC markedly elevated {metric} ({describe(metric, 'MC')}) versus NC "
            f"{describe(metric, 'NC')} ({format_p_value(p_val)})."
        )

    for metric in ("IL_1B", "IL_18", "TNFA", "CRP"):
        for group in ("LD", "HD"):
            p_val, _ = get_pair_result(stats[metric]["pairs"], "MC", group)
            highlights.append(
                f"{group} reduced {metric} to {describe(metric, group)}, "
                f"{format_p_value(p_val)} vs MC."
            )

    record_results_section(
        "Figure 4 - Inflammatory Cytokines",
        highlights,
        methods_note="Mean ± SD, one-way ANOVA plus Tukey HSD focused on MC comparisons.",
    )

    record_figure_legend(
        "Fig4_Inflammatory_Markers",
        "Immunofluorescence quantification of inflammatory mediators.",
        panels=[
            ("A", "IL-1β concentrations."),
            ("B", "IL-18 concentrations."),
            ("C", "TNF-α concentrations."),
            ("D", "CRP levels."),
        ],
        stats_note="Bars represent mean ± SD. Tukey HSD comparisons vs MC (* p<0.05, ** p<0.01, *** p<0.001).",
    )


def main():
    df = create_if_dataframe()
    df_long, summary = build_long_format(df)
    stats = run_statistics(df_long)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_if_panels(df_long, stats)
    figure_paths = export_figure(fig, "Fig4_Inflammatory_Markers")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
