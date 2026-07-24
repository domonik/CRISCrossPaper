import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, SINGLE_COL, PT7
from helpers import add_bar_values
from plotly.subplots import make_subplots

fixNames = {
    "CRISPROfft": "CRISPR-OFFT",
    "CRISPert_Tcell_PT": "CRISPert",
    "cnnCRISPR": "CnnCrispr",
    "crisprIP": "CRISPR-IP",
    "CRISPert": "CRISPert", 
    "CRISCross": "CRISCross",
    "CRISCrossThree": "CRISCrossThree"
    
}

fix_celltype = {
    "u20s": "U2OS",
    "hek293t": "HEK293T",
    "k562": "K562"
}

def main():
    cols = ["K562"]
    df = pd.read_csv("../Results/combined_cross_cell_results_3cell.csv")
    origdf = pd.read_csv("../datasets/k562_deepcrispr_withCoords_hg38.csv")
    baseline_random_chance = origdf["label"].mean()
    file = "../Results/FinalSummary.tsv"
    df2 = pd.read_csv(file, sep="\t")
    agstring = "Sequence & AlphaGenome ATAC-seq"
    seed_cut = [42 * i for i in range(50)][25]
    breakpoint()

    df2 = df2[
        (df2["dataset"] == "K562") & (df2["mode"] == "Arti-Ws512-AG_ccTesting") |
        (df2["dataset"] == "K562") & (df2["mode"] == "Arti-Ws512+AG_ccTesting") & (df2["run"].str.contains("AG")) |
        (df2["dataset"] == "K562") & (df2["mode"] == "Arti-Ws23-AG_ccTesting")  |
        (df2["dataset"] == "K562") & (df2["mode"] == "Arti-Ws23+AG_ccTesting") & (df2["run"].str.contains("AG")) 
    ]
    breakpoint()

    df2 = df2[(df2["run"].str[-1] == "0")]

    df2 = df2[df2["seed"] < seed_cut]

    df2["name"] = "CRISCross"
    df2.loc[df2["comparison"] == "ThreeTrack", "name"] = "CRISCrossThree"
    df2["aucpr"] = df2["AUCPR"]
    df2["cell_type"] = df2["dataset"]
    df2["features"] = df2["run"].apply(lambda x: agstring if "+AG" in x else "Sequence only")
    df["cell_type"] = df["cell_type"].map(fix_celltype)
    df = df[df["cell_type"] == "K562"]
    df["features"] = "Sequence only"
    breakpoint()
    df = pd.concat((df, df2))

    fig = make_subplots(
        rows=1, cols=len(df["cell_type"].unique()),
        #column_titles=cols,
        shared_yaxes=True
        #column_widths=[0.6, 0.4]
    )
    fig.update_annotations(font=dict(size=PT7))
    df = df.groupby(["name", "cell_type", "features", "window_size"], as_index=False, dropna=False)["aucpr"].mean()   
    df = df.sort_values("aucpr")
    df["model"] = df["name"].map(fixNames)
    df.loc[df["model"] == "CRISCross", "model"] =  df.loc[df["model"] == "CRISCross"]["model"] + "<br>" + df.loc[df["model"] == "CRISCross"]["window_size"].astype(int).astype(str) + " nt"

    for i, cell_type in enumerate(cols):
        sdf = df[df["cell_type"] == cell_type]
        for feat in ["Sequence only", agstring]:
            sdf2 = sdf[sdf["features"] == feat]
            color = COLORS["jaxgold"] if "ATAC" in feat else COLORS["seagrey"]
            if "ATAC" in feat:
                breakpoint()
            fig.add_trace(
                    go.Bar(
                        x=sdf2["model"],
                        y=sdf2["aucpr"],
                        #error_y=dict(type="data", array=df_ws["std"]),
                        marker=dict(color=color, line=dict(color="black", width=1)),
                        showlegend=True,
                        name=feat,
                        width=0.4
                        
                    ),
                    col=i+1,
                    row=1
            )
    fig.update_layout(
        template="simple_white_custom",
        yaxis=dict(title="AUC-PR"),
        width = SINGLE_COL,
        height=175,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(r=20, b=10)
        
    )
    fig.add_hline(
        y=baseline_random_chance,
        line=dict(dash="dash", color="black")
    )
    print(f"Random chance: {baseline_random_chance}")
    fig.add_annotation(
        text=f"{baseline_random_chance:.3f}",
        showarrow=False,
        x=1,
        xanchor="right",
        y=baseline_random_chance,
        yanchor="bottom",
        yref="y",
        xref="x domain",
        textangle=270
        
    )
    
    df["mean"] = df["aucpr"]
    add_bar_values(fig, df, modes=df["model"].tolist(), x_val="model", offset_x=0)
    fig.write_html("Figures/CrossCellTypeOldModels.html")
    fig.write_image("Figures/CrossCellTypeOldModels.svg")



if __name__ == "__main__":
    main()