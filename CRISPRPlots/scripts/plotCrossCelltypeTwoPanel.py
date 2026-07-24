import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, SINGLE_COL, PT7
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

cellTypes = ["K562", "IPSC"]
agstring = "Sequence & AlphaGenome ATAC-seq"
modes_filter = ["Arti-Ws512-AG_ccTesting", "Arti-Ws512+AG_ccTesting"]
old_models = ["CRISPROfft", "CRISPert", "cnnCRISPR", "crisprIP"]


def main():
    file = "../Results/FinalSummary.tsv"
    df2 = pd.read_csv(file, sep="\t")

    # --- CRISCross rows ---
    cross = df2[(df2["dataset"].isin(cellTypes)) & (df2["mode"].isin(modes_filter))]
    cross = cross[cross["run"].str[-1] == "0"]
    cross = cross[cross["seed"] < 1050]
    cross["name"] = "CRISCross"
    cross["aucpr"] = cross["AUCPR"]
    cross["cell_type"] = cross["dataset"]
    cross["features"] = cross["mode"].apply(
        lambda m: agstring if "+AG" in m else "Sequence only"
    )

    # --- old models ---
    old = df2[
        (df2["dataset"].isin(cellTypes)) &
        (df2["model"].isin(old_models)) &
        (df2["mode"] == "Base")
    ]
    old["name"] = old["model"]
    old["aucpr"] = old["AUCPR"]
    old["cell_type"] = old["dataset"]
    old["features"] = "Sequence only"

    # Combine and aggregate
    df = pd.concat((old, cross))
    df = df.groupby(["name", "cell_type", "features"],
                     as_index=False, dropna=False)["aucpr"].mean()

    # Assign a unique numeric x-position to every row per cell type
    # so no two bars share a tick and barmode grouping is irrelevant.
    order = [agstring, "Sequence only"]  # ATAC first, then seq within same model
    pos_map = {
        ("CRISCross", agstring): 0,
        ("CRISCross", "Sequence only"): 1,
        ("CRISPROfft", "Sequence only"): 2,
        ("CRISPert", "Sequence only"): 3,
        ("cnnCRISPR", "Sequence only"): 4,
        ("crisprIP", "Sequence only"): 5,
    }
    tick_labels = {
        0: "CRISCross",
        1: "CRISCross",
        2: "CRISPR-OFFT",
        3: "CRISPert",
        4: "CnnCrispr",
        5: "CRISPR-IP",
    }

    df["xpos"] = df.apply(lambda r: pos_map[(r["name"], r["features"])], axis=1)
    df["_color"] = df["features"].apply(
        lambda f: COLORS["jaxgold"] if "ATAC" in f else COLORS["seagrey"]
    )

    # Build figure
    fig = make_subplots(
        rows=1, cols=len(cellTypes),
        horizontal_spacing=0.05,
        shared_yaxes=True
    )
    fig.update_annotations(font=dict(size=PT7))

    added_legends = set()

    for i, ct in enumerate(cellTypes):
        sdf = df[df["cell_type"] == ct]
        for _, row in sdf.iterrows():
            showlegend = row["features"] not in added_legends
            added_legends.add(row["features"])
            fig.add_trace(
                go.Bar(
                    x=[row["xpos"]],
                    y=[row["aucpr"]],
                    marker=dict(color=row["_color"],
                                line=dict(color="black", width=1)),
                    showlegend=showlegend,
                    name=row["features"],
                    width=0.4
                ),
                col=i + 1, row=1
            )

    # Set custom tick labels on both x-axes
    tick_vals = list(tick_labels.keys())
    tick_txt = [tick_labels[t] for t in tick_vals]

    fig.update_xaxes(
        tickvals=tick_vals, ticktext=tick_txt,
        title_text="K562", col=1
    )
    fig.update_xaxes(
        tickvals=tick_vals, ticktext=tick_txt,
        title_text="IPSC", col=2
    )

    fig.update_layout(
        template="simple_white_custom",
        yaxis=dict(title="AUC-PR", range=[0, 1.15]),
        width=WIDTH,
        height=200,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(r=20, b=10),
    )

    # Baselines — each line only in its own panel
    baseline_k562 = 0.00641
    baseline_ipsc = 1 / 40

    fig.add_shape(
        type="line", x0=0, y0=baseline_k562, x1=1, y1=baseline_k562,
        xref="x domain", yref="y",
        line=dict(dash="dash", color="black", width=2),
    )
    fig.add_annotation(
        text=f"{baseline_k562:.3f}", showarrow=False,
        x=1, xanchor="right", y=baseline_k562, yanchor="bottom",
        yref="y", xref="x domain", textangle=270,
        row=1, col=1
    )

    fig.add_shape(
        type="line", x0=0, y0=baseline_ipsc, x1=1, y1=baseline_ipsc,
        xref="x2 domain", yref="y",
        line=dict(dash="dash", color="black", width=2),
    )
    fig.add_annotation(
        text=f"{baseline_ipsc:.3f}", showarrow=False,
        x=1, xanchor="right", y=baseline_ipsc, yanchor="bottom",
        yref="y", xref="x2 domain", textangle=270,
        row=1, col=2
    )

    # Value labels on bars — 90 degree angle
    for i, ct in enumerate(cellTypes):
        sdf = df[df["cell_type"] == ct]
        for _, row in sdf.iterrows():
            fig.add_annotation(
                x=row["xpos"],
                y=row["aucpr"],
                text=f"{row['aucpr']:.3f}",
                showarrow=False,
                xref=f"x{i+1}",
                yref="y",
                yanchor="bottom",
                xanchor="center",
                textangle=90,
                font=dict(size=PT7)
            )

    fig.write_html("Figures/CrossCelltypeTwoPanel.html")
    fig.write_image("Figures/CrossCelltypeTwoPanel.svg")


if __name__ == "__main__":
    main()
