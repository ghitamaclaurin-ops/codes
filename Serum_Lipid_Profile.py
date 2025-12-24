# -*- coding: utf-8 -*-
"""
Serum lipid profile analysis for NC, MC, LD, and HD groups.

Outputs:
- lipid_profile_data.csv / summary.csv / stats.xlsx
- Fig2_Serum_Lipid.svg / pdf (2×2 panels with Tukey annotations vs MC)
- README + figure legend + results summary entries
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

PARAMETERS = ["TC", "TG", "LDL", "HDL"]
PANEL_LABELS = ["A", "B", "C", "D"]
PANEL_TITLES = {
    "TC": "Total cholesterol",
    "TG": "Triglycerides",
    "LDL": "LDL-C",
    "HDL": "HDL-C",
}
AXIS_LABELS = {
    "TC": "Total cholesterol (mmol/L)",
    "TG": "Triglycerides (mmol/L)",
    "LDL": "LDL-C (mmol/L)",
    "HDL": "HDL-C (mmol/L)",
}

# Data loaded from CSV file
df = pd.read_csv("data/lipid_data.csv")


def create_lipid_data() -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv("data/lipid_data.csv")


def build_long_format(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        for metric in PARAMETERS:
            rows.append({
                "group": row["Group"],
                "metric": metric,
                "value": row[metric],
                "unit": "mmol/L",
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
    for metric in PARAMETERS:
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
    raw_csv = OUTDIR / "lipid_profile_data.csv"
    sum_csv = OUTDIR / "lipid_profile_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)

    xlsx_path = OUTDIR / "lipid_profile_stats.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for param, result in stats.items():
            result["anova"].to_excel(writer, sheet_name=f"{param}_ANOVA")
            result["tukey"].to_excel(writer, sheet_name=f"{param}_Tukey", index=False)
    return raw_csv, sum_csv, xlsx_path


def plot_lipid_panels(df_long: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))
    axes = axes.flatten()

    for idx, param in enumerate(PARAMETERS):
        ax = axes[idx]
        sub = df_long[df_long["metric"] == param]
        means = [sub[sub["group"] == g]["value"].mean() for g in GROUP_ORDER]
        sds = [sub[sub["group"] == g]["value"].std() for g in GROUP_ORDER]
        render_bar_panel(
            ax,
            panel_label=PANEL_LABELS[idx],
            panel_title=PANEL_TITLES[param],
            groups=GROUP_ORDER,
            means=means,
            sds=sds,
            tukey_pairs=stats[param]["pairs"],
            ylabel=AXIS_LABELS[param],
        )

    plt.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def write_readme(raw_csv, sum_csv, xlsx_path, figure_paths):
    content = dedent(f"""
    血脂分析结果
    ============

    文件说明:
      - lipid_profile_data.csv         原始血脂数据
      - lipid_profile_summary.csv      各指标均值 ± SD
      - lipid_profile_stats.xlsx       One-way ANOVA + Tukey HSD
      - Fig2_Serum_Lipid.svg / pdf     四面板血脂图（灰度 + 显著性）

    文件路径:
      - 数据: {raw_csv.name}, {sum_csv.name}
      - 统计: {xlsx_path.name}
      - 图像: {figure_paths['svg'].name}, {figure_paths['pdf'].name}
    """).strip()
    write_text_file("README_lipid_profile.txt", content)


def summarize_results(summary: pd.DataFrame, stats: dict):
    indexed = summary.set_index(["metric", "group"])

    def describe(metric, group):
        row = indexed.loc[(metric, group)]
        return f"{row['mean']:.2f} ± {row['sd']:.2f}"

    highlights = []

    # MC burden vs NC
    for metric in ("TC", "TG", "LDL"):
        p_val, _ = get_pair_result(stats[metric]["pairs"], "MC", "NC")
        highlights.append(
            f"MC exhibited elevated {metric} ({describe(metric, 'MC')}) compared with NC "
            f"{describe(metric, 'NC')} ({format_p_value(p_val)})."
        )

    # Intervention effects vs MC
    for metric in ("TC", "TG", "LDL"):
        for group in ("LD", "HD"):
            p_val, _ = get_pair_result(stats[metric]["pairs"], "MC", group)
            highlights.append(
                f"{group} lowered {metric} to {describe(metric, group)} vs MC {describe(metric, 'MC')} "
                f"({format_p_value(p_val)})."
            )

    # HDL restoration
    for group in ("LD", "HD"):
        p_val, _ = get_pair_result(stats["HDL"]["pairs"], "MC", group)
        highlights.append(
            f"{group} restored HDL levels ({describe('HDL', group)}) relative to MC "
            f"{describe('HDL', 'MC')} ({format_p_value(p_val)})."
        )

    record_results_section(
        "Figure 2 - Serum Lipids",
        highlights,
        methods_note="Mean ± SD, one-way ANOVA plus Tukey HSD annotated vs MC.",
    )

    record_figure_legend(
        "Fig2_Serum_Lipid",
        "Serum lipid profile modulation across treatment groups.",
        panels=[
            ("A", "Total cholesterol."),
            ("B", "Triglycerides."),
            ("C", "Low-density lipoprotein cholesterol."),
            ("D", "High-density lipoprotein cholesterol."),
        ],
        stats_note="Bars represent mean ± SD. Tukey HSD comparisons against MC (*/**/***).",
    )


def main():
    df = create_lipid_data()
    df_long, summary = build_long_format(df)
    stats = run_statistics(df_long)
    raw_csv, sum_csv, xlsx_path = save_tables(df, summary, stats)

    fig = plot_lipid_panels(df_long, stats)
    figure_paths = export_figure(fig, "Fig2_Serum_Lipid")
    plt.close(fig)

    write_readme(raw_csv, sum_csv, xlsx_path, figure_paths)
    summarize_results(summary, stats)

    print("Saved:", raw_csv, sum_csv, xlsx_path, figure_paths["svg"], figure_paths["pdf"])


if __name__ == "__main__":
    main()
