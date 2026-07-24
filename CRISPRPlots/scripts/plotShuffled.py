"""
Comparison plots for AGTensorsWrongCellType, ShuffledAGTensors, and the
original AG mode (all at window_size == 23).

AG is the reference; the other two are negative-control conditions (wrong
cell type AG tensors, and shuffled AG tensors), pooled across pretrain,
model, and seed — one row per split per mode after averaging any duplicate
raw rows. Three figures are produced, reusing the same helpers/colors/
statistics as plotCrossAttnResults_v3.py:

  1. AUCPR              - three-bar comparison, each mode Wilcoxon-tested
                           against AG, with a significance bracket
  2. Precision@k        - three-line comparison across k = 10/50/100/500/1000
  3. Recall@90precision - three-bar comparison, same significance testing
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotCrossAttentionResultsV4 import (
    COLORS,
    PREC_COLS,
    RECALL90_COL,
    _fix_and_merge_90precision_cols,
    add_significance_brackets,
    calc_k_stats,
    calc_sig_ablation,
    stars_label,
)

MODE_A = "AGTensorsWrongCellType"
MODE_B = "ShuffledAGTensors"
MODE_C = "-Histone"

# AG (the real/original tensors) is the reference bar/line; A and B are the
# two negative controls tested against it.
MODE_ORDER = [MODE_C, MODE_A, MODE_B]
REFERENCE_MODE = MODE_C

GROUP_COLORS = {
    MODE_A: COLORS["jaxpetrol"],
    MODE_B: COLORS["seagrey"],
    MODE_C: COLORS["jaxgold"],
}
GROUP_LABELS = {
    MODE_A: "AG tensors (wrong cell type)",
    MODE_B: "Shuffled AG tensors",
    MODE_C: "Original AG",
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _prep_split_level(df):
    """
    Filter to the three comparison modes at window_size == 23 and collapse
    to one row per (mode, split), averaging over pretrain, model, seed, and
    any other duplicate raw rows.
    """
    df = _fix_and_merge_90precision_cols(df)
    sub = df[
        df["mode"].isin([MODE_A, MODE_B, MODE_C]) & (df["window_size"] == 23)
    ].copy()
    value_cols = (
        ["AUCPR"]
        + [c for c in PREC_COLS if c in sub.columns]
        + ([RECALL90_COL] if RECALL90_COL in sub.columns else [])
    )
    sub = sub.groupby(["mode", "split"])[value_cols].mean().reset_index()
    return sub


def calc_sig(sub, value_col, reference_mode=REFERENCE_MODE):
    """Wilcoxon test of every non-reference mode vs reference_mode, paired
    by split, BH-corrected across the modes tested. Reuses the same
    reference-vs-group logic as the ablation panel in
    plotCrossAttnResults_v3.py."""
    return calc_sig_ablation(sub, reference_mode=reference_mode, value_col=value_col)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_bar_comparison(
    sub, value_col, y_label, modes_order=MODE_ORDER, reference_mode=REFERENCE_MODE,
):
    """Bar comparison across modes_order, with each non-reference mode
    Wilcoxon-tested (paired by split) against reference_mode and a
    significance bracket drawn between it and the reference bar."""
    agg = (
        sub.groupby("mode")[value_col]
        .agg(["mean", "std"])
        .reindex(modes_order)
    )
    sig_df = calc_sig(sub, value_col, reference_mode=reference_mode)
    print(sig_df)
    fig = go.Figure()
    for mode in modes_order:
        if mode not in agg.index or pd.isna(agg.loc[mode, "mean"]):
            continue
        fig.add_trace(go.Bar(
            x=[GROUP_LABELS[mode]], y=[agg.loc[mode, "mean"]],
            error_y=dict(type="data", array=[agg.loc[mode, "std"]], thickness=1),
            marker=dict(color=GROUP_COLORS[mode], line=dict(color="black", width=1)),
            width=0.4, showlegend=False,
        ))

    ref_x = modes_order.index(reference_mode)
    comps = []
    for mode in modes_order:
        if mode == reference_mode:
            continue
        prow = sig_df[sig_df["mode"] == mode]
        if prow.empty:
            continue
        label = stars_label(prow["padj"].values[0])
        comps.append((ref_x, modes_order.index(mode), label))

    y_max = agg["mean"].max()
    if comps and not np.isnan(y_max):
        add_significance_brackets(
            fig, comps, y_start=y_max + 0.1, y_step=0.1, row=1, col=1, ncols=1,
        )
        y_range = [0, y_max + 0.1 + 0.1 * len(comps) + 0.15]
    else:
        y_range = [0, max(1.15, (y_max + 0.3) if not np.isnan(y_max) else 1.15)]

    fig.update_layout(
        template="simple_white_custom",
        height=280, width=420,
        margin=dict(l=60, t=25, r=20),
    )
    fig.update_yaxes(title=dict(text=y_label), range=y_range)
    return fig


def make_precision_k_comparison(sub, modes_order=MODE_ORDER):
    """Line comparison across modes_order, mean over splits with std error
    bars at each k."""
    fig = go.Figure()
    breakpoint()
    for mode in modes_order:
        mode_data = sub[sub["mode"] == mode]
        if mode_data.empty:
            continue
        k_values, means, stds = calc_k_stats(mode_data)
        if not k_values:
            continue
        fig.add_trace(go.Scatter(
            x=k_values, y=means,
            error_y=dict(type="data", array=stds, thickness=1),
            mode="lines+markers",
            name=GROUP_LABELS[mode],
            line=dict(color=GROUP_COLORS[mode], width=1.5),
            marker=dict(color=GROUP_COLORS[mode], size=6, line=dict(color="black", width=1)),
        ))

    k_tickvals = [int(c.split("@")[1]) for c in PREC_COLS]
    fig.update_xaxes(type="linear", tickvals=k_tickvals, title=dict(text="k"))
    fig.update_yaxes(title=dict(text="Precision@k"))
    fig.update_layout(
        template="simple_white_custom",
        height=280, width=450,
        margin=dict(l=60, t=25, r=20),
        legend=dict(title=dict(text="")),
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    sub = _prep_split_level(df)

    if sub.empty:
        print(f"No rows found for modes {MODE_A!r} / {MODE_B!r} / {MODE_C!r} at window_size 23.")
        return

    # -- AUCPR --
    fig_aucpr = make_bar_comparison(sub, "AUCPR", "AUC-PR")
    fig_aucpr.write_html("Figures/AGShuffleComparison_AUCPR.html")
    fig_aucpr.write_image(
        "Figures/AGShuffleComparison_AUCPR.svg",
        width=fig_aucpr.layout.width, height=fig_aucpr.layout.height,
    )
    print("AUCPR comparison saved.")

    # -- Precision@k --
    have_prec_cols = all(c in sub.columns for c in PREC_COLS)
    if have_prec_cols:
        fig_prec = make_precision_k_comparison(sub)
        fig_prec.write_html("Figures/AGShuffleComparison_PrecAtK.html")
        fig_prec.write_image(
            "Figures/AGShuffleComparison_PrecAtK.svg",
            width=fig_prec.layout.width, height=fig_prec.layout.height,
        )
        print("Precision@k comparison saved.")
    else:
        print("Skipping precision@k comparison - missing precision@k columns.")

    # -- Recall@90precision --
    if RECALL90_COL in sub.columns:
        fig_recall = make_bar_comparison(sub, RECALL90_COL, "Recall@90%-precision")
        fig_recall.write_html("Figures/AGShuffleComparison_Recall90.html")
        fig_recall.write_image(
            "Figures/AGShuffleComparison_Recall90.svg",
            width=fig_recall.layout.width, height=fig_recall.layout.height,
        )
        print("Recall@90precision comparison saved.")
    else:
        print(f"Skipping Recall@90precision comparison - missing {RECALL90_COL}.")


if __name__ == "__main__":
    main()