# -*- coding: utf-8 -*-
"""
Body weight trajectory analysis.

The script reproduces the descriptive statistics, one-way ANOVA, Tukey HSD,
and multi-panel publication figure requested for the Inflammation manuscript.
Running the script writes all outputs into ``figures_results`` relative to the
project root.
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

METRICS = ["Initial_Weight", "Final_Weight", "Weight_Gain"]
PANEL_LABELS = ["A", "B", "C"]
PANEL_TITLES = {
    "Initial_Weight": "Initial body weight",
    "Final_Weight": "Final body weight",
    "Weight_Gain": "Body weight gain",
}
YLABELS = {
    "Initial_Weight": "Initial body weight (g)",
    "Final_Weight": "Final body weight (g)",
    "Weight_Gain": "Weight gain (g)",
}

# Data loaded from CSV file
df_weight = pd.read_csv("data/weight_data.csv")


def create_dataframe() -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv("data/weight_data.csv")


def build_long_format(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        for metric in METRICS:
            rows.append({
                "group": row["Group"],
                "treatment": row["Treatment"],
                "metric": metric,
                "value": row[metric],
                "unit": "g",
            })

    df_long = pd.DataFrame(rows)
    summary = (
        df_long.groupby(["metric", "group"])
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
        .reset_index()
    )
    summary["mean_sd"] = summary["mean"].round(2).astype(str) + " ± " + summary["sd"].round(2).astype(str)
    return df_long, summary


def run_anova_and_tukey(df_long: pd.DataFrame):
    stats = {}
    for metric in METRICS:
        sub = df_long[df_long["metric"] == metric].copy()
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
    raw_csv = OUTDIR / "weight_change_data.csv"
    sum_csv = OUTDIR / "weight_change_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "weight_change_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for metric, comp in stats.items():
            comp["anova"].to_excel(writer, sheet_name=f"{metric}_ANOVA")
            comp["tukey"].to_excel(writer, sheet_name=f"{metric}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_weight_panels(df_long: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(1, 3, figsize=panel_figsize(1, 3))
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
    体重变化分析结果
    ================

    文件说明:
      - weight_change_data.csv         原始体重数据
      - weight_change_summary.csv      均值 ± 标准差
      - weight_change_stats.xlsx       One-way ANOVA + Tukey HSD
      - Fig1_Weight_Gain.svg / pdf     3面板体重图（灰度 + 显著性）

    结果概述:
      - 模型组（MC）初始及最终体重均显著高于NC
      - 药物干预（LD / HD）显著降低了体重增量
      - 图中标注的 * / ** / *** 均来自Tukey HSD对MC的比较

    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_weight_change.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    indexed = summary.set_index(["metric", "group"])

    def mean_sd(metric, group):
        row = indexed.loc[(metric, group)]
        return row["mean"], row["sd"]

    def describe(metric, group):
        m, sd = mean_sd(metric, group)
        return f"{m:.2f} ± {sd:.2f} g"

    highlights = []

    # Initial weight MC vs NC
    p_val, _ = get_pair_result(stats["Initial_Weight"]["pairs"], "MC", "NC")
    highlights.append(
        f"MC animals began heavier ({describe('Initial_Weight', 'MC')}) than NC ({describe('Initial_Weight', 'NC')}), "
        f"{format_p_value(p_val)}."
    )

    # Final weight reductions in LD/HD vs MC
    for group in ("LD", "HD"):
        p_val, _ = get_pair_result(stats["Final_Weight"]["pairs"], "MC", group)
        highlights.append(
            f"{group} dosing lowered final weight to {describe('Final_Weight', group)} vs MC {describe('Final_Weight', 'MC')} "
            f"({format_p_value(p_val)})."
        )

    # Weight gain improvements
    for group in ("LD", "HD"):
        p_val, _ = get_pair_result(stats["Weight_Gain"]["pairs"], "MC", group)
        highlights.append(
            f"{group} mitigated weight gain ({describe('Weight_Gain', group)}) relative to MC "
            f"{describe('Weight_Gain', 'MC')} ({format_p_value(p_val)})."
        )

    record_results_section(
        "Figure 1 - Body Weight",
        highlights,
        methods_note="Mean ± SD, one-way ANOVA with Tukey HSD; annotations reference MC comparisons.",
    )

    record_figure_legend(
        "Fig1_Weight_Gain",
        "Body weight dynamics across the NC, MC, LD, and HD groups.",
        panels=[
            ("A", "Baseline body mass after acclimation."),
            ("B", "Terminal body mass following intervention."),
            ("C", "Absolute weight gain calculated per subject."),
        ],
        stats_note="Bars show mean ± SD. Significance from Tukey HSD vs MC (* p<0.05, ** p<0.01, *** p<0.001).",
    )


def main():
    df = create_dataframe()
    df_long, summary = build_long_format(df)
    stats = run_anova_and_tukey(df_long)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_weight_panels(df_long, stats)
    figure_paths = export_figure(fig, "Fig1_Weight_Gain")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
