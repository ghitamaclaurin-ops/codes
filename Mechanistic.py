# -*- coding: utf-8 -*-
"""
Simulated mechanistic endpoints (NF-κB activation, ROS, macrophage polarization).

Generates reproducible synthetic data, tables, and a six-panel grayscale figure
corresponding to Figure 7 in the manuscript.
"""

from __future__ import annotations

import numpy as np
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

np.random.seed(42)

BASE_PARAMETERS = {
    "p_p65_ratio": {"unit": "a.u.", "NC": (1.00, 0.15), "MC": (2.20, 0.25), "LD": (1.60, 0.20), "HD": (1.20, 0.15)},
    "IkBa_rel": {"unit": "a.u.", "NC": (1.00, 0.12), "MC": (0.55, 0.10), "LD": (0.80, 0.12), "HD": (0.95, 0.10)},
    "ROS_intensity": {"unit": "a.u.", "NC": (100, 12), "MC": (180, 18), "LD": (145, 15), "HD": (115, 12)},
    "M1_percent": {"unit": "%", "NC": (35, 6), "MC": (70, 7), "LD": (55, 6), "HD": (40, 6)},
}

METRIC_ORDER = ["p_p65_ratio", "IkBa_rel", "ROS_intensity", "M1_percent", "M2_percent", "M1_to_M2_ratio"]
PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]
PANEL_TITLES = {
    "p_p65_ratio": "p-p65 / total p65",
    "IkBa_rel": "IκBα abundance",
    "ROS_intensity": "Reactive oxygen species",
    "M1_percent": "M1 macrophages",
    "M2_percent": "M2 macrophages",
    "M1_to_M2_ratio": "M1 vs M2 balance",
}
YLABELS = {
    "p_p65_ratio": "p-p65 / total p65 (a.u.)",
    "IkBa_rel": "IκBα (a.u.)",
    "ROS_intensity": "ROS intensity (a.u.)",
    "M1_percent": "M1 macrophages (%)",
    "M2_percent": "M2 macrophages (%)",
    "M1_to_M2_ratio": "M1/M2 ratio",
}


def clamp(values, low=None, high=None):
    arr = np.asarray(values, dtype=float)
    if low is not None:
        arr = np.maximum(arr, low)
    if high is not None:
        arr = np.minimum(arr, high)
    return arr


def simulate_dataset(n_per_group: int = 6) -> pd.DataFrame:
    """Load pre-simulated data from CSV file."""
    return pd.read_csv("data/mechanistic_data.csv")


def summarize(df: pd.DataFrame):
    summary = (
        df.groupby(["metric", "group"], observed=True)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
        .reset_index()
    )
    summary["mean_sd"] = summary["mean"].round(2).astype(str) + " ± " + summary["sd"].round(2).astype(str)
    return summary


def run_statistics(df: pd.DataFrame):
    stats = {}
    for metric in METRIC_ORDER:
        sub = df[df["metric"] == metric]
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
    raw_csv = OUTDIR / "mechanistic_simulated_raw.csv"
    sum_csv = OUTDIR / "mechanistic_simulated_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "mechanistic_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for metric, result in stats.items():
            result["anova"].to_excel(writer, sheet_name=f"{metric}_ANOVA")
            result["tukey"].to_excel(writer, sheet_name=f"{metric}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_mechanistic_panels(df: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(2, 3, figsize=panel_figsize(2, 3))
    axes = axes.flatten()

    for idx, metric in enumerate(METRIC_ORDER):
        ax = axes[idx]
        sub = df[df["metric"] == metric]
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
    NF-κB / ROS / Macrophage模拟分析
    ==============================

    文件说明:
      - mechanistic_simulated_raw.csv     个体模拟数据
      - mechanistic_simulated_summary.csv 均值 ± SD
      - mechanistic_stats.xlsx            One-way ANOVA + Tukey HSD
      - Fig7_Mechanistic_Assessments.*    6面板机制图
    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_simulation.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    indexed = summary.set_index(["metric", "group"])

    def describe(metric, group):
        row = indexed.loc[(metric, group)]
        return f"{row['mean']:.2f} ± {row['sd']:.2f}"

    highlights = []
    key_metrics = ["p_p65_ratio", "IkBa_rel", "ROS_intensity", "M1_percent", "M2_percent", "M1_to_M2_ratio"]
    for metric in key_metrics:
        for group in ("LD", "HD"):
            p_val, _ = get_pair_result(stats[metric]["pairs"], "MC", group)
            highlights.append(
                f"{group} shifted {metric} to {describe(metric, group)} vs MC "
                f"{describe(metric, 'MC')} ({format_p_value(p_val)})."
            )

    record_results_section(
        "Figure 7 - Mechanistic Readouts",
        highlights,
        methods_note="Simulated mean ± SD (n=6/group). One-way ANOVA with Tukey HSD vs MC.",
    )

    record_figure_legend(
        "Fig7_Mechanistic_Assessments",
        "Simulated NF-κB/ROS/macrophage endpoints under NC, MC, LD, and HD conditions.",
        panels=[
            ("A", "p-p65 to total p65 ratio."),
            ("B", "IκBα relative levels."),
            ("C", "ROS intensity."),
            ("D", "M1 macrophage percentage."),
            ("E", "M2 macrophage percentage."),
            ("F", "M1/M2 ratio."),
        ],
        stats_note="Bars show mean ± SD. Tukey HSD annotations vs MC (* p<0.05, ** p<0.01, *** p<0.001).",
    )


def main():
    df = simulate_dataset()
    summary = summarize(df)
    stats = run_statistics(df)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_mechanistic_panels(df, stats)
    figure_paths = export_figure(fig, "Fig7_Mechanistic_Assessments")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
