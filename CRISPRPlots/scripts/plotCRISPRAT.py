import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon, mannwhitneyu
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
    row=1,
    col=1,
    ncols=3,
    tick_frac=0.35,
    font_size_stars=12,
):
    """
    Draw stacked horizontal significance brackets above grouped bars.
    """
    idx = (row - 1) * ncols + col
    xref = "x" if idx == 1 else f"x{idx}"
    yref = "y" if idx == 1 else f"y{idx}"
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
            x=(x0 + x1) / 2, y=y,
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
    sdf = df[(df["mode"] == "Base") & (df["window_size"].isin([512]))]
    cdf = df[(df["comparison"] == "CrossCellType") & (df["window_size"].isin([512]))]
    cdf = cdf.sort_values("seed")
    cdf = cdf[cdf["mode"].str.contains("Testing")]
    pcdf = cdf.groupby(["mode", "dataset", "split", "window_size"])["AUCPR"].agg(['mean', 'std']).reset_index()
    ag_res = mannwhitneyu(cdf[cdf["mode"] == "Arti-Ws512-AG_ccTesting"]["AUCPR"], cdf[cdf["mode"] == "Arti-Ws512+AG_ccTesting"]["AUCPR"])
    print(ag_res)
    print(pcdf)
    breakpoint()


    df =  df[(df["comparison"] == "CRISPRAT") & (df["window_size"].isin([512]))]
    #df = pd.concat((df, sdf))
    df = df.groupby(["mode", "dataset", "split", "window_size"])["AUCPR"].mean().reset_index()
    df = df.sort_values("split")


    df["pretrain strategy"] = df["mode"].apply(lambda x: "CRISPRAT" if x.split("-")[0] == "Arti" and "-Epi" in x else "CRISPRAT+Epi" if x.split("-")[0] == "Arti" and "+Epi" in x else "CRISPert")
    datasets = ["T-Cell", "K562"]
    res_dfs = []
    for dataset in datasets:
        strategies = df[df["dataset"] == dataset]["pretrain strategy"].unique()
        strategy_pairs = list(combinations(strategies, 2))  # all-vs-all pairs
        print(strategies)
        pvals = []

        for s1, s2 in strategy_pairs:
            df1 = df[(df["pretrain strategy"] == s1) & (df["dataset"] == dataset)]["AUCPR"]
            df2 = df[(df["pretrain strategy"] == s2) & (df["dataset"] == dataset)]["AUCPR"]
            if len(df1) and len(df2):
                res = wilcoxon(df1, df2)
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
            "features 1": ["Sequence" for _ in strategy_pairs],
            "features 2": ["Sequence" for _ in strategy_pairs],
            "fineTuningDataset": dataset
        })

        res_dfs.append(results_df)
        
    row = {
        "Strategy 1": "CRISPRAT",
        "Strategy 2": "CRISPRAT",
        "p-value": ag_res.pvalue,
        "padj": ag_res.pvalue,
        "test dataset": "K562",
        "features 1": "Sequence",
        "features 2": "Sequence & ATAC-seq",
        "fineTuningDataset": "T-Cell"
    }
    results_df = pd.concat(res_dfs)
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    print(results_df)
    df = df.groupby(["mode", "pretrain strategy", "dataset", "window_size"])["AUCPR"].agg(['mean', 'std']).reset_index()
    

    
    df["window_size"] = df["window_size"].astype(str)
    # Get unique facets and colors
    window_sizes = df["window_size"].unique()

    # Create a subplot for each dataset
    fig = make_subplots(
        cols=1, rows=len(datasets) + 1,
        row_titles=["T-Cell<br>(leave one out)", "K562<br>(leave one out)", "K562<br>(Cross cell type)"],
        vertical_spacing=0.05
        #column_widths=[0.6, 0.4]
    )
    fig.update_annotations(font=dict(size=PT6))
    for anno in fig.layout.annotations:
        if anno.text.startswith("pretrain"):
            anno.update(y=anno.y+0.15)
    cmap = {
        "CRISPert": COLORS["seagrey"],
        "CRISPRAT": COLORS["jaxgold"],
        "CRISPRAT+Epi": COLORS["seagrey"],
    }
    hatch_pattern = {
        "CRISPert": None,
        "CRISPRAT": "/",
        "CRISPRAT+Epi": "/",
    }
    for col_index, dataset in enumerate(datasets, start=1):
        df_sub = df[df["dataset"] == dataset]
        for ws in window_sizes:
            df_ws = df_sub[df_sub["window_size"] == ws]
            for pretrain_strat in df_ws["pretrain strategy"].unique():
                df_ws_p  = df_ws[df_ws["pretrain strategy"] == pretrain_strat]
                fig.add_trace(
                    go.Bar(
                        x=df_ws_p["pretrain strategy"],
                        y=df_ws_p["mean"],
                        name=f"{ws}",
                        #error_y=dict(type="data", array=df_ws["std"]),
                        marker=dict(
                            color=COLORS["jaxgold"] if "Epi" in pretrain_strat else COLORS["seagrey"],
                            line=dict(color="black", width=1),
                            pattern=dict(shape=hatch_pattern[pretrain_strat] or "")
                        ),
                        showlegend=False,
                        width=0.4
                    ),
                    col=1,
                    row=col_index
                )
    pcdf["Feature"] = pcdf["mode"].apply(lambda x: "Sequence &<br>ATAC-seq" if "+AG" in x else "Sequence")
    # Color based on x-axis value: ATAC-seq → jaxgold, else → seagrey
    # CRISPRAT trained bars (the ATAC-seq one) should have lines pattern
    def get_bar_color(feature):
        if "ATAC-seq" in feature:
            return COLORS["jaxgold"]
        return COLORS["seagrey"]


    for _, row_pcdf in pcdf.iterrows():
        feature = row_pcdf["Feature"]
        fig.add_trace(
            go.Bar(
                x=[feature],
                y=[row_pcdf["mean"]],
                marker=dict(
                    color=get_bar_color(feature),
                    line=dict(color="black", width=1),
                    pattern=dict(shape="/")
                ),
                showlegend=False,
                width=0.4
            ),
            col=1,
            row=3
        )  
                
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape=hatch_pattern["CRISPert"] or "")
            ),
            name="Pretraining on<br>T-cell dataset"

        )
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape=hatch_pattern["CRISPRAT"] or "")
            ),
            name="CRISPRAT"

        )
    )
    x_order = ["CRISPert", "CRISPRAT", "CRISPRAT+Epi"]
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=x_order,
        row=1,
        col=1,
        tickvals=x_order,                  # original values
        ticktext=[ "Sequence +<br>EX Epi" if "+Epi" in m else "Sequence" for m in x_order ]
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=x_order[:2],
            tickvals=x_order,                  # original values
        ticktext=[ "Sequence +<br>EX Epi" if "+Epi" in m else "Sequence" for m in x_order ],        
        col=1,
        row=2
    )
    cros_cell_order = [ "Sequence", "Sequence &<br>ATAC-seq"]
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=cros_cell_order,
            tickvals=cros_cell_order,                  # original values
        col=1,
        row=3,
    )
    fig.update_traces(error_y=dict(thickness=1))

    tcell_df=df[df["dataset"] == "T-Cell"]
    sub_res_df = results_df[(results_df["Strategy 2"] == "CRISPert") | (results_df["Strategy 1"] == "CRISPert")]
    tcell_results = sub_res_df[sub_res_df["test dataset"] == "T-Cell"]

    # Build comparisons for row 1 (T-Cell)
    strategy_to_x = {s: i for i, s in enumerate(x_order)}
    comps_row1 = []
    for _, row in tcell_results.iterrows():
        s1 = row["Strategy 1"]
        s2 = row["Strategy 2"]
        pval = row["padj"]
        if np.isnan(pval):
            continue
        label = stars_label(pval)
        x0 = strategy_to_x[s1]
        x1 = strategy_to_x[s2]
        comps_row1.append((x0, x1, label))

    if comps_row1:
        y_max_row1 = tcell_df["mean"].max() 
        add_significance_brackets(
            fig, comps_row1,
            y_start=y_max_row1 + 0.3,
            y_step=0.05,
            row=1, col=1, ncols=1,
        )

    k562_df = df[df["dataset"] == "K562"]
    k562_results = sub_res_df[sub_res_df["test dataset"] == "K562"]
    comps_row2 = []
    for _, row in k562_results.iterrows():
        s1 = row["Strategy 1"]
        s2 = row["Strategy 2"]
        pval = row["padj"]
        if np.isnan(pval):
            continue
        label = stars_label(pval)
        x0 = strategy_to_x[s1]
        x1 = strategy_to_x[s2]
        comps_row2.append((x0, x1, label))

    if comps_row2:
        y_max_row2 = k562_df["mean"].max() 
        add_significance_brackets(
            fig, comps_row2,
            y_start=y_max_row2 + 0.26,
            y_step=0.05,
            row=2, col=1, ncols=1,
        )

    add_bar_values(fig, tcell_df, modes=x_order, col=1, row=1, x_val="pretrain strategy",)
    add_bar_values(fig, k562_df, modes=x_order, col=1, row=2, x_val="pretrain strategy",)

    # Row 3: Cross cell type comparison (Sequence vs Sequence & ATAC-seq)
    cross_cell_results = results_df[(results_df["Strategy 2"] == "CRISPRAT") & (results_df["Strategy 1"] == "CRISPRAT")]
    feature_to_x = {s: i for i, s in enumerate(cros_cell_order)}
    comps_row3 = []
    for _, row in cross_cell_results.iterrows():
        s1 = row["features 1"]
        s2 = row["features 2"]
        s1 = 'Sequence &<br>ATAC-seq' if "ATAC" in s1 else s1
        s2 = 'Sequence &<br>ATAC-seq' if "ATAC" in s2 else s2
        pval = row["padj"]
        if np.isnan(pval):
            continue
        label = stars_label(pval)
        x0 = feature_to_x[s1]
        x1 = feature_to_x[s2]
        comps_row3.append((x0, x1, label))

    if comps_row3:
        y_max_row3 = pcdf["mean"].max() 
        add_significance_brackets(
            fig, comps_row3,
            y_start=y_max_row3 + 0.3,
            y_step=0.05,
            row=3, col=1, ncols=1,
        )

    add_bar_values(fig, pcdf, modes=cros_cell_order, col=1, row=3, x_val="Feature")


    fig.update_layout(
        barmode="stack",
        template="simple_white_custom",
        height=500,
        width = 150,
        margin=dict(t=20, l=30, r=10),
        legend=dict(orientation="h", y=-0.05)
    )
    fig.update_yaxes(title_text="AUC-PR", col=1,)
    fig.update_yaxes(range=[0,1.15])
    fig.write_html("Figures/CRISPRAT.html")
    fig.write_image("Figures/CRISPRAT.svg", width=fig.layout.width, height=fig.layout.height)
    

    
    
if __name__ == "__main__":
    main()