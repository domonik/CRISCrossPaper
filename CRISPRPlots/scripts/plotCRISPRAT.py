import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon, mannwhitneyu
import plotly.express as px
from plotly_template import COLORS, WIDTH, SINGLE_COL, PT6
import plotly.graph_objects as go
from helpers import add_bar_values, add_stars_with_stripes, calc_sig
from plotly.subplots import make_subplots
import numpy as np
from itertools import combinations




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

    df =  df[(df["comparison"] == "CRISPRAT") & (df["window_size"].isin([512]))]
    #df = pd.concat((df, sdf))
    df = df.groupby(["mode", "dataset", "split", "window_size"])["AUCPR"].mean().reset_index()
    df = df.sort_values("split")


    df["pretrain strategy"] = df["mode"].apply(lambda x: "CRISPRAT" if x.split("-")[0] == "Arti" and "-Epi" in x else "CRISPRAT+Epi" if x.split("-")[0] == "Arti" and "+Epi" in x else "CRISPert")
    breakpoint()
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
            breakpoint()
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
        "CRISPert": COLORS["jaxpetrol"],
        "CRISPRAT": COLORS["jaxgold"],
        "CRISPRAT+Epi": COLORS["jaxgold"],
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
                        marker=dict(color=cmap[pretrain_strat], line=dict(color="black", width=1)),
                        showlegend=False,
                        width=0.4
                    ),
                    col=1,
                    row=col_index
                )
    pcdf["Feature"] = pcdf["mode"].apply(lambda x: "Sequence &<br>ATAC-seq" if "+AG" in x else "Sequence")
    fig.add_trace(
        go.Bar(
            x=pcdf["Feature"],
            y=pcdf["mean"],
            #error_y=dict(type="data", array=pcdf["std"]),
            marker=dict(color=cmap["CRISPRAT"], line=dict(color="black", width=1)),
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
            marker=dict(color=cmap["CRISPert"], line=dict(color="black", width=1)),
            name="Pretraining on<br>T-cell dataset"

        )
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color=cmap["CRISPRAT"], line=dict(color="black", width=1)),
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
        row=3
    )
    fig.update_traces(error_y=dict(thickness=1))

    tcell_df=df[df["dataset"] == "T-Cell"]
    sub_res_df = results_df[(results_df["Strategy 2"] == "CRISPert") | (results_df["Strategy 1"] == "CRISPert")]
    add_stars_with_stripes(sub_res_df[sub_res_df["test dataset"] == "T-Cell"], strategy_order=x_order, tcell_df=df[df["dataset"] == "T-Cell"], fig=fig, subplot_idx=1, space_fac=1.1)
    add_stars_with_stripes(sub_res_df[sub_res_df["test dataset"] == "K562"], strategy_order=x_order[:2], tcell_df=df[df["dataset"] == "K562"], fig=fig, subplot_idx=2)
    #add_stars_with_stripes(results_df[results_df["test dataset"] == "K562"], strategy_order=x_order[:2], tcell_df=df[df["dataset"] == "K562"], fig=fig, subplot_idx=2)
    add_bar_values(fig, tcell_df, modes=x_order, col=1, row=1, x_val="pretrain strategy",)
    tcell_df=df[df["dataset"] == "K562"]
    add_bar_values(fig, tcell_df, modes=x_order, col=1, row=2, x_val="pretrain strategy",)
    
    sub_res_df = results_df[(results_df["Strategy 2"] == "CRISPRAT") & (results_df["Strategy 1"] == "CRISPRAT")]
    add_stars_with_stripes(sub_res_df, strategy_order=[c.replace("<br>", " ") for c in cros_cell_order], tcell_df=pcdf, fig=fig, subplot_idx=3, strategy_col_prefix="features", space_fac=1.1)
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
    fig.write_html("Figures/CRISPRAT.html")
    fig.write_image("Figures/CRISPRAT.svg", width=fig.layout.width, height=fig.layout.height)
    

    
    
if __name__ == "__main__":
    main()