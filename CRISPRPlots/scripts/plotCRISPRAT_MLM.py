import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
import plotly.express as px
from plotly_template import COLORS, WIDTH, SINGLE_COL, PT6
import plotly.graph_objects as go
from helpers import add_bar_values, calc_sig
from plotly.subplots import make_subplots
import numpy as np
from itertools import combinations


def add_significance_brackets(
    fig,
    comparisons,
    y_start,
    y_step,
    xref,
    yref,
    font_size_stars=12,
    tick_frac=0.35,
):
    """
    Draw stacked horizontal significance brackets above grouped bars.
    """
    tick_h = y_step * tick_frac
    sorted_comps = sorted(comparisons, key=lambda c: abs(c[1] - c[0]))

    for i, (x0, x1, label) in enumerate(sorted_comps):
        y = y_start + i * y_step

        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        fig.add_shape(
            type="line", x0=x0, x1=x0, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        fig.add_shape(
            type="line", x0=x1, x1=x1, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        is_stars = "*" in label
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y - 0.06*is_stars,
            yanchor="bottom", xanchor="center",
            text=label, showarrow=False,
            xref=xref, yref=yref,
            font=dict(size=font_size_stars if is_stars else PT6),
        )


def stars_label(p):
    """Returns star string or 'n.s.' – always produces a label."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")

    # ── Row 1 & 2: CRISPRAT leave-one-out ──────────────────────────
    crisprat_df = df[(df["comparison"] == "CRISPRAT") & (df["window_size"].isin([512]))]
    crisprat_df = crisprat_df.sort_values("split")

    # Derive pretrain strategy from mode
    crisprat_df["pretrain strategy"] = crisprat_df["mode"].apply(
        lambda x: (
            "CRISPRAT"
            if x.split("-")[0] == "Arti" and "-Epi" in x
            else "CRISPRAT+Epi"
            if x.split("-")[0] == "Arti" and "+Epi" in x
            else "CRISPert"
        )
    )

    datasets = ["T-Cell", "K562"]
    res_dfs = []
    for dataset in datasets:
        df_ds = crisprat_df[crisprat_df["dataset"] == dataset]
        strategies = df_ds["pretrain strategy"].unique()
        strategy_pairs = list(combinations(strategies, 2))
        pvals = []

        for s1, s2 in strategy_pairs:
            df1 = df_ds[df_ds["pretrain strategy"] == s1]["AUCPR"]
            df2 = df_ds[df_ds["pretrain strategy"] == s2]["AUCPR"]
            if len(df1) and len(df2):
                res = mannwhitneyu(df1, df2)
                pvals.append(res.pvalue)
            else:
                pvals.append(np.nan)

        # Correct p-values for non-NaN entries
        pvals_array = np.array(pvals, dtype=float)
        mask = ~np.isnan(pvals_array)
        pvals_corrected = np.full_like(pvals_array, np.nan)
        pvals_corrected[mask] = multipletests(pvals_array[mask], method="bonferroni")[1]

        results_df = pd.DataFrame({
            "Strategy 1": [s1 for s1, s2 in strategy_pairs],
            "Strategy 2": [s2 for s1, s2 in strategy_pairs],
            "p-value": pvals,
            "padj": pvals_corrected,
            "test dataset": dataset,
        })
        res_dfs.append(results_df)

    all_results_df = pd.concat(res_dfs, ignore_index=True)

    # Aggregate for bar charts (mean over splits × seeds)
    crisprat_agg = (
        crisprat_df
        .groupby(["mode", "pretrain strategy", "dataset", "window_size"])["AUCPR"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # ── Row 3: CrossCellType + CROSSCellMLM ────────────────────────
    # CrossCellType data (Arti models, ws=512)
    cdf = df[(df["comparison"] == "CrossCellType") & (df["window_size"].isin([512]))]
    cdf = cdf.sort_values("seed")
    cdf = cdf[cdf["mode"].str.contains("Testing")]

    # CROSSCellMLM data (MLM models, ws=512)
    mlm_df = df[(df["comparison"] == "CROSSCellMLM") & (df["window_size"].isin([512]))]
    mlm_df = mlm_df.sort_values("seed")

    # Combine CrossCellType and CROSSCellMLM for row 3
    combined_raw = pd.concat([cdf, mlm_df], ignore_index=True)

    # Aggregate per mode
    cross_agg = (
        combined_raw
        .groupby(["mode", "dataset", "split", "window_size"])["AUCPR"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Define row 3 bar order, labels, colors, and hatch patterns
    row3_modes = [
        "Arti-Ws512-AG_ccTesting",
        "Arti-Ws512+AG_ccTesting",
        "MLMFull-Ws512-AG_ccTesting",
        "MLMFull-Ws512+AG_ccTesting",
        "MLMTCell-Ws512-AG_ccTesting",
        "MLMTCell-Ws512+AG_ccTesting",
    ]

    row3_labels = [
        "CRISPRAT\nSequence",
        "CRISPRAT\nSequence+ATAC",
        "MLMFull\nSequence",
        "MLMFull\nSequence+ATAC",
        "MLMTCell\nSequence",
        "MLMTCell\nSequence+ATAC",
    ]

    row3_colors = [
        COLORS["seagrey"],       # Arti-AG
        COLORS["jaxgold"],       # Arti+AG
        COLORS["seagrey"],       # MLMFull-AG
        COLORS["jaxgold"],       # MLMFull+AG
        COLORS["seagrey"],       # MLMTCell-AG
        COLORS["jaxgold"],       # MLMTCell+AG
    ]

    row3_hatches = [
        "/",    # Arti-AG — CRISPRAT hatch
        "/",    # Arti+AG — CRISPRAT hatch
        "+",    # MLMFull-AG — MLM hatch
        "+",    # MLMFull+AG — MLM hatch
        "",    # MLMTCell-AG — MLM hatch
        "",    # MLMTCell+AG — MLM hatch
    ]

    # ── Significance tests for row 3 ───────────────────────────────
    # All comparisons are against Arti-Ws512+AG_ccTesting
    arti_plus_ag = combined_raw[combined_raw["mode"] == "Arti-Ws512+AG_ccTesting"]["AUCPR"]

    # 1) Arti-AG vs Arti+AG (within CrossCellType)
    arti_minus_ag = combined_raw[combined_raw["mode"] == "Arti-Ws512-AG_ccTesting"]["AUCPR"]
    res_arti = mannwhitneyu(arti_minus_ag, arti_plus_ag)

    # 2-5) Each MLM mode vs Arti+AG
    mlm_pvals = []
    for m in row3_modes[2:]:  # skip first 2 (Arti modes)
        mlm_data = combined_raw[combined_raw["mode"] == m]["AUCPR"]
        if len(mlm_data) > 0:
            res = mannwhitneyu(mlm_data, arti_plus_ag)
            mlm_pvals.append(res.pvalue)
        else:
            mlm_pvals.append(np.nan)

    # Bonferroni correction across all row-3 tests
    all_row3_pvals = [res_arti.pvalue] + mlm_pvals
    _, all_row3_padj, _, _ = multipletests(all_row3_pvals, method="bonferroni")

    print("=== Row 3 significance (vs Arti+AG) ===")
    print(f"Arti-AG vs Arti+AG: p={res_arti.pvalue:.2e}, padj={all_row3_padj[0]:.2e}")
    for i, m in enumerate(row3_modes[2:]):
        print(f"{m} vs Arti+AG: p={mlm_pvals[i]:.2e}, padj={all_row3_padj[i+1]:.2e}")

    # ── Build figure ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],

        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    fig.update_annotations(font=dict(size=PT6))
    for anno in fig.layout.annotations:
        if anno.text.startswith("T-Cell") or anno.text.startswith("Cross-cell"):
            anno.update(y=anno.y + 0.12)

    # ── Row 1: T-Cell ──────────────────────────────────────────────
    tcell_df = crisprat_agg[crisprat_agg["dataset"] == "T-Cell"]
    x_order_r1 = ["CRISPert", "CRISPRAT", "CRISPRAT+Epi"]
    hatch_r1 = {
        "CRISPert": "",
        "CRISPRAT": "/",
        "CRISPRAT+Epi": "/",
    }

    for pretrain_strat in x_order_r1:
        bar = tcell_df[tcell_df["pretrain strategy"] == pretrain_strat]
        if bar.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[pretrain_strat],
                y=[bar["mean"].values[0]],
                marker=dict(
                    color=COLORS["jaxgold"] if "Epi" in pretrain_strat else COLORS["seagrey"],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=hatch_r1[pretrain_strat]),
                ),
                showlegend=False,
                width=0.4,
            ),
            row=1, col=1,
        )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=x_order_r1,
        tickvals=x_order_r1,
        ticktext=[
            "Sequence" if s == "CRISPert" else
            "Sequence" if s == "CRISPRAT" else
            "Sequence +<br>EX Epi"
            for s in x_order_r1
        ],
        row=1, col=1,
    )

    # Significance brackets row 1
    sub_res_df = all_results_df[
        (all_results_df["Strategy 2"] == "CRISPert") |
        (all_results_df["Strategy 1"] == "CRISPert")
    ]
    tcell_results = sub_res_df[sub_res_df["test dataset"] == "T-Cell"]
    strategy_to_x = {s: i for i, s in enumerate(x_order_r1)}

    comps_r1 = []
    for _, row in tcell_results.iterrows():
        s1, s2 = row["Strategy 1"], row["Strategy 2"]
        pval = row["padj"]
        if np.isnan(pval):
            continue
        comps_r1.append((strategy_to_x[s1], strategy_to_x[s2], stars_label(pval)))

    if comps_r1:
        y_max_r1 = tcell_df["mean"].max()
        add_significance_brackets(
            fig, comps_r1,
            y_start=y_max_r1 + 0.15,
            y_step=0.04,
            xref="x", yref="y",
        )

    # Bar value annotations row 1
    for i, strat in enumerate(x_order_r1):
        bar = tcell_df[tcell_df["pretrain strategy"] == strat]
        if bar.empty:
            continue
        fig.add_annotation(
            x=strat, y=bar["mean"].values[0],
            text=f"{bar['mean'].values[0]:.3f}",
            showarrow=False, xref="x", yref="y",
            row=1, col=1, yanchor="bottom", xanchor="center",
            font=dict(size=PT6, color="black"),
        )

    # ── Row 1, Col 2: K562 ─────────────────────────────────────────
    k562_df = crisprat_agg[(crisprat_agg["dataset"] == "K562") & (~crisprat_agg["mode"].str.contains("+AG", regex=False))]
    # K562 may only have CRISPert and CRISPRAT (no CRISPRAT+Epi)
    x_order_r2 = [s for s in x_order_r1 if not k562_df[k562_df["pretrain strategy"] == s].empty]
    for pretrain_strat in x_order_r2:
        bar = k562_df[k562_df["pretrain strategy"] == pretrain_strat]
        if bar.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[pretrain_strat],
                y=[bar["mean"].values[0]],
                marker=dict(
                    color=COLORS["jaxgold"] if "Epi" in pretrain_strat else COLORS["seagrey"],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=hatch_r1[pretrain_strat]),
                ),
                showlegend=False,
                width=0.4,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=x_order_r2,
        tickvals=x_order_r2,
        ticktext=[
            "Sequence" if s == "CRISPert" else
            "Sequence" if s == "CRISPRAT" else
            "Sequence +<br>EX Epi"
            for s in x_order_r2
        ],
        row=1, col=2,
    )

    # Significance brackets row 2 (K562)
    k562_results = sub_res_df[sub_res_df["test dataset"] == "K562"]
    comps_r2 = []
    for _, row in k562_results.iterrows():
        s1, s2 = row["Strategy 1"], row["Strategy 2"]
        pval = row["padj"]
        if np.isnan(pval):
            continue
        comps_r2.append((strategy_to_x[s1], strategy_to_x[s2], stars_label(pval)))

    if comps_r2:
        y_max_r2 = k562_df["mean"].max()
        add_significance_brackets(
            fig, comps_r2,
            y_start=y_max_r2 + 0.15,
            y_step=0.05,
            xref="x2", yref="y2",
        )

    # Bar value annotations row 2
    for i, strat in enumerate(x_order_r2):
        bar = k562_df[k562_df["pretrain strategy"] == strat]
        if bar.empty:
            continue
        fig.add_annotation(
            x=strat, y=bar["mean"].values[0],
            text=f"{bar['mean'].values[0]:.3f}",
            showarrow=False, xref="x2", yref="y2",
            row=1, col=2, yanchor="bottom", xanchor="center",
            font=dict(size=PT6, color="black"),
        )

    # ── Row 2 (spanned): Cross-cell type + CROSSCellMLM ────────────
    # Filter aggregated data to only include our defined modes
    cross_plot = cross_agg[cross_agg["mode"].isin(row3_modes)].copy()

    for idx, mode in enumerate(row3_modes):
        bar = cross_plot[cross_plot["mode"] == mode]
        if bar.empty:
            # Add empty bar if mode has no data
            fig.add_trace(
                go.Bar(
                    x=[row3_labels[idx]],
                    y=[0],
                    marker=dict(
                        color=row3_colors[idx],
                        line=dict(color="black", width=1),
                        pattern=dict(shape=row3_hatches[idx]),
                    ),
                    showlegend=False,
                    width=0.4,
                ),
                row=2, col=1,
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=[row3_labels[idx]],
                    y=[bar["mean"].values[0]],
                    marker=dict(
                        color=row3_colors[idx],
                        line=dict(color="black", width=1),
                        pattern=dict(shape=row3_hatches[idx]),
                    ),
                    showlegend=False,
                    width=0.4,
                ),
                row=2, col=1,
            )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=row3_labels,
        tickvals=row3_labels,
        ticktext=row3_labels,
        row=2, col=1,
    )

    # Significance brackets row 3
    # Arti-AG vs Arti+AG → bars 0 vs 1
    # MLM modes (bars 2-5) vs Arti+AG (bar 1)
    comps_r3 = []
    # Arti-AG vs Arti+AG
    if not np.isnan(all_row3_padj[0]):
        comps_r3.append((0, 1, stars_label(all_row3_padj[0])))
    # Each MLM mode vs Arti+AG (bar index 1)
    for i, m in enumerate(row3_modes[2:]):
        padj = all_row3_padj[i + 1]
        if not np.isnan(padj):
            comps_r3.append((i + 2, 1, stars_label(padj)))

    if comps_r3:
        y_max_r3 = cross_plot["mean"].max()
        n_brackets = len(comps_r3)
        add_significance_brackets(
            fig, comps_r3,
            y_start=y_max_r3 + 0.15,
            y_step=0.08,
            xref="x3", yref="y3",
        )

    # Bar value annotations row 3
    for idx, mode in enumerate(row3_modes):
        bar = cross_plot[cross_plot["mode"] == mode]
        if bar.empty:
            continue
        fig.add_annotation(
            x=row3_labels[idx], y=bar["mean"].values[0],
            text=f"{bar['mean'].values[0]:.3f}",
            showarrow=False, xref="x3", yref="y3",
            row=2, col=1, yanchor="bottom", xanchor="center",
            font=dict(size=PT6, color="black"),
        )

    # ── Legend entries (invisible bars) ─────────────────────────────
    row3_text = [
        "Sequence",
        "Sequence &<br>ATAC",
        "Sequence",
        "Sequence &<br>ATAC",
        "Sequence",
        "Sequence &<br>ATAC",
    ]
    fig.update_xaxes(
        categoryorder="array", categoryarray=row3_labels,
        tickvals=row3_labels,
        ticktext=row3_text,
        row=2, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape="/"),
            ),
            name="CRISPRAT",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(
                color="black",
                line=dict(color="black", width=1),
                pattern=dict(shape=""),
            ),
            name="Pretraining on<br>T-cell dataset",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape="+"),
            ),
            name="Pretraining on<br>T-cell & K562 dataset",
        )
    )


    # ── Layout ──────────────────────────────────────────────────────
    fig.update_layout(
        barmode="group",
        template="simple_white_custom",
        height=400,
        width=SINGLE_COL,
        margin=dict(t=20, l=30, r=10, b=50),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )

    fig.update_yaxes(title_text="AUC-PR", row=1, col=1)
    fig.update_yaxes(title_text="AUC-PR", row=1, col=2)
    fig.update_yaxes(title_text="AUC-PR", row=2, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_yaxes(range=[0, 1.15], row=2, col=1)

    # Remove x-axis titles for clean look
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=2)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    
    
    fig.add_annotation(
        text="T-Cell<br>(leave one out)",
        x=0.5,
        xanchor="center",
        xref="x domain",
        y=1.1,
        yref="y domain",
        showarrow=False
    )
        
    fig.add_annotation(
        text="K562<br>(leave one out)",
        x=0.5,
        xanchor="center",
        xref="x2 domain",
        y=1.1,
        yref="y2 domain",
        showarrow=False
    )
    fig.add_annotation(
        text="K562<br>(Cross cell type)",
        x=0.5,
        xanchor="center",
        xref="x3 domain",
        y=1.0,
        yref="y3 domain",
        showarrow=False
    )


    fig.write_html("Figures/CRISPRATAblationRevision.html")
    fig.write_image("Figures/CRISPRATAblationRevision.svg", width=fig.layout.width, height=fig.layout.height)
    print("Saved Figures/CRISPRAT_MLM.html and Figures/CRISPRAT_MLM.svg")


if __name__ == "__main__":
    main()
