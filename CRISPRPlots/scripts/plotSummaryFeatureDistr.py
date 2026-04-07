import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.express as px
from statsmodels.stats.multitest import multipletests
import numpy as np
from gatherDistributions import FEATURES
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from  plotly_template import COLORS, WIDTH, rgb_to_rgba, PT7
from sklearn.preprocessing import StandardScaler

EPI_FEATURES = [
            "ATAC",
            "EX_ATAC",
            "EX_H3K4me1",
            "EX_H3K4me3",
            "EX_H3K9me3",
            "EX_H3K27ac",
            "EX_H3K27me3",
            "EX_H3K36me3",
            'H3K27ac',
            'H3K27me3',
            'H3K36me3',
            'H3K4me1',
            'H3K4me3',
            'H3K9me3',
]

def main():
    df = pd.read_csv("SummaryStatsPerWinsize.tsv", sep="\t")
    base = {f.replace("EX_", "") for f in EPI_FEATURES}

    # keep only features where both raw and EX exist
    valid = {f for f in base if f in EPI_FEATURES and f"EX_{f}" in EPI_FEATURES}

    # columns to keep
    cols_to_keep = [
        c for c in df.columns
        if any(v in c for v in valid)
    ] + ["chr", "start", "end",  "Strand", "Guide_sequence", "ID", "label", "AlphagenomeIndex"]
    df = df[cols_to_keep]
    feat_cols = [col for col in df.columns if col not in ["chr", "start", "end",  "Strand", "Guide_sequence", "ID", "label", "AlphagenomeIndex"]]
    results = []
    for col in feat_cols:
        group1 = df.loc[df["label"] == 1, col].dropna()
        group0 = df.loc[df["label"] == 0, col].dropna()
        n1 = len(group1)
        n0 = len(group0)
        
        stat, pval = mannwhitneyu(group0, group1, alternative="two-sided")
        r_rb = 1 - (2 * stat) / (n1 * n0)
        
        results.append({
            "feature": col,
            "U_statistic": stat,
            "p_value": pval,
            "rank_biserial": r_rb
        })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("rank_biserial", ascending=False).reset_index()
    
    results_df["window_size"] = results_df["feature"].str.split("_").str[-1].astype(int)
    results_df["Type"] = "AG"
    mask = results_df["feature"].str.startswith("EX")
    results_df.loc[mask, "Type"] = "EX"
    results_df.loc[mask, "Feature"] = results_df.loc[mask, "feature"].str.split("_").str[1]
    results_df.loc[~mask, "Feature"] = results_df.loc[~mask, "feature"].str.split("_").str[0]
    results_df.loc[results_df["Feature"] == "+", "Feature"] = "RNAseq +"
    results_df.loc[results_df["Feature"] == "-", "Feature"] = "RNAseq -"
    results_df["mode"] = results_df["feature"].str.split("_").str[-2]
    results_df["feature_type"] = results_df["feature"].apply(lambda x: "Experimental" if x.startswith("EX") else "AlphaGenome")
    
    _, results_df["padj"], _, _ = multipletests(results_df["p_value"])
    results_df["significant"] = results_df["padj"] <= 0.05
    results_df["effect_size"] = results_df["rank_biserial"].abs()

    results_df = results_df.sort_values(["significant", "effect_size"], ascending=False)
    results_df["Rank"] = np.arange(len(results_df)) + 1
    opdf = results_df[["Rank", "Feature", "mode", "feature_type", "window_size", "rank_biserial", "p_value", "padj", "significant"]].copy()
    for col in ["p_value", "padj"]:
        opdf[col] = opdf[col].map(lambda x: f"{x:.3e}")
    other_float_cols = opdf.select_dtypes(include="float").columns.difference(["p_value", "padj"])
    opdf[other_float_cols] = opdf[other_float_cols].round(3)
    opdf.to_csv("Tables/FeatureMWUTable.tsv", sep="\t", index=False)
    results_df = results_df[results_df["significant"]]

    
    best_ex = results_df[results_df["feature_type"] == "Experimental"].iloc[0]["feature"]
    best_ag = results_df[results_df["feature_type"] == "AlphaGenome"].iloc[0]["feature"]
    quantiles = np.quantile(df[best_ag], np.linspace(0, 1, 50))  # 50 bins by quantiles
# assign each value to a bin
    df["bin"] = np.digitize(df[best_ag], quantiles, right=True)

    # optional: create bin labels for plotting
    df["bin_center"] = [quantiles[i-1] + (quantiles[i]-quantiles[i-1])/2 if i>0 else quantiles[0] for i in df["bin"]]
    #df[best_ag] = np.log10(df[best_ag])
    #df[best_ex] = np.log10(df[best_ex])
    # histogram

    fig = make_subplots(rows=2, cols=2, column_titles=["Experimental", "AlphaGenome"])
    fig.update_annotations(font=dict(size=PT7))
    fig.update_layout(
        template="simple_white_custom"
    )
    cmap = {
        1: COLORS["jaxpetrol"], 
        0: COLORS["jaxgold"]
    }
    # helper to add histogram for a column
    scaler = StandardScaler()
    for row, col in enumerate((best_ag, best_ex)):
        if col.startswith("EX"):
            col = col[3:]
        ag_name = col
        ex_name = f"EX_{col}"
        values = df[ag_name].values.reshape(-1, 1)
        if ag_name.startswith("H3"):
            df[ag_name] = np.log10(values)
            pre, sub = "log<sub>10</sub>(", ")"

        else:
            df[ag_name] = np.log10(values)
            pre, sub = "log<sub>10</sub>(", ")"
        #df[ex_name] = np.log1p(df[ex_name].values.reshape(-1, 1))

        bin_edges_ag = np.linspace(df[ag_name].min(), df[ag_name].max(), 51)
        bin_edges_ex = np.linspace(df[ex_name].min(), df[ex_name].max(), 51)
        for lidx, label in enumerate(df["label"].unique()):
            sdf = df[df["label"] == label]
            if row == 0:
                name="GUIDE-seq" if label else "Cas-OFFfinder"
            else:
                name = None
            

            fig.add_trace(
                go.Histogram(
                    x=sdf[ex_name],
                    xbins=dict(start=bin_edges_ag[0], end=bin_edges_ex[-1], size=bin_edges_ex[1]-bin_edges_ex[0]),
                    marker=dict(color=cmap[label], line=dict(color="black", width=1)),
                    histnorm="probability",
                    name=name,
                    showlegend=True if name else False

                    ),
                row=row+1, col=1)
            fig.add_trace(
                go.Histogram(
                    x=sdf[ag_name],
                    xbins=dict(start=bin_edges_ag[0], end=bin_edges_ag[-1], size=bin_edges_ag[1]-bin_edges_ag[0]),
                    marker=dict(color=cmap[label], line=dict(color="black", width=1)),
                    histnorm="probability",
                    name=None,
                    showlegend=False

                    
                    ),
                row=row+1, col=2)
        fig.update_xaxes(title_text=f"{pre}{ag_name.replace("_", " ")} nt{sub}", row=row+1, col=2)
        fig.update_xaxes(title_text=ex_name[3:].replace("_", " ") + " nt", row=row+1, col=1)

    fig.update_traces(opacity=0.7)
    fig.update_layout(barmode="overlay", height=250, width = int(WIDTH / 2), template="simple_white_custom", 
        legend=dict(orientation="h", yanchor="top", y=-0.15, yref="paper"
                    ),
        margin=dict(r=10, t=10)
                      )
    fig.write_html("Figures/SummaryDistribution.html")
    fig.write_image("Figures/SummaryDistribution.svg")
    

    

    

    idx = results_df.groupby(["Feature", "Type"])["effect_size"].idxmax()
    
    rdf = (results_df.loc[idx, ["Feature", "Type", "effect_size", "window_size", "mode"]].reset_index(drop=True))
    rdf["window_size"] = rdf["window_size"].astype(int)
    cmap = {
        "EX": COLORS["jaxpetrol"], 
        "AG": COLORS["jaxgold"]
    }
    fig = go.Figure()

    types = rdf["Type"].unique()

    for i, t in enumerate(types):
        d = rdf[rdf["Type"] == t]

        fig.add_bar(
            x=d["Feature"],
            y=d["window_size"],
            name=t,
            showlegend=False,
            marker=dict(
                color=d["effect_size"],
                coloraxis=f"coloraxis{i+1}",
                line=dict(color="black", width=1)
            ),
        )
    rmin = results_df["effect_size"].min()
    rmax = results_df["effect_size"].max()
    # Define two independent color scales
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[rmin],
                coloraxis="coloraxis3"
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        marker=dict(
            color=cmap["AG"],
            symbol="square",
            line=dict(color="black", width=1)
        ),
        mode="markers",
        showlegend=True,
        name="AlphaGenome"
        
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        marker=dict(
            color=cmap["EX"],
            symbol="square",
            line=dict(color="black", width=1)
        ),
        mode="markers",
        showlegend=True,
        name="Experimental",
    )
    fig.update_layout(
        barmode="group",
        yaxis_type="log",
        template="simple_white_custom",
        bargap=0.2,
        bargroupgap=0.2,
        height=250,
        width=int(WIDTH / 2) - 20,
        margin=dict(r=20, l=50),
        coloraxis=dict(
            showscale=True,
            cmin=rmin, cmax=rmax,
            colorscale=[rgb_to_rgba(cmap["AG"], 0), cmap["AG"]],
            colorbar=dict(
                title=dict(text=""),
                ticklen=0,
                ticklabelposition="outside left",
                showticklabels=False,
                # remove numbers from first bar
                x=1.05, xanchor="left",
                len=0.7, y=0.7, yanchor="top", thickness=15,
            )
        ),
        coloraxis2=dict(
            showscale=True,
            cmin=rmin, cmax=rmax,
            colorscale=[rgb_to_rgba(cmap["EX"], 0), cmap["EX"]],
            colorbar=dict(
                title=dict(text=""),
                x=1.15, xanchor="left",
                len=0.7, y=0.7, yanchor="top", thickness=15,
            )
        ),
        coloraxis3=dict(showscale=False),   # hide the old one
    )
    desired_order = (
    rdf.sort_values("window_size")["Feature"]
       .tolist()
    )
    fig.add_annotation(
        text="|Rank biserial|",
        x=1.05, xref="paper", xanchor="left",
        y=0.67, yref="paper", yanchor="bottom",  # just above the colorbar top
        showarrow=False,
        font=dict(size=PT7),
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=desired_order
    )
    fig.update_yaxes(title=dict(text="Context size [nt]"), title_standoff=10)
    fig.write_html("Figures/RankBiserialPlot.html")
    fig.write_image("Figures/RankBiserialPlot.svg")
    
    


if __name__ == "__main__":
    main()