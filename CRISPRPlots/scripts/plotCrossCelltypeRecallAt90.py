import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, PT7
from plotly.subplots import make_subplots

AG_STRING = "Sequence & AlphaGenome ATAC-seq"

# "warm" uses the warm_axis palette below; "cold" swaps in the cold palette
# for every model other than CRISCross, which keeps its own colors either way.
COLOR_SCHEME = "cold"

# Same layout/coloring as row 1 of plotCrossCelltypeThreePanelPerGuide.py,
# but plotting recall@90%precision instead of AUC-PR.

SERIES_ORDER = [
    "CRISCross_AG",
    "CRISCross_seq",
    "CRISPR-OFFT",
    "CRISPert",
    "CnnCrispr",
    "CRISPR-IP",
    "CRISOT",
    "CrisprBERT",
]

# warm_axis <- c("#D3BC8D","#FFA300","#C83803","#97233F","#34302B","#311D00")
# cold_axis <- c("#69BE28","#4B92DB","#008E97","#125740","#003594","#002244")
# CRISCross keeps its own coloring (gold for +AG, seagrey for seq-only) in
# either scheme; the warm/cold palette covers the other six models.
NON_CRISCROSS_PALETTES = {
    "warm": ["#D3BC8D", "#FFA300", "#C83803", "#97233F", "#34302B", "#311D00"],
    "cold": ["#6B8EAD", "#A3BE8C", "#B48EAD", "#8CCEE0", "#D08770 ", "#8FBCBB"],
}
NON_CRISCROSS_MODELS = [
    "CRISOT", "CRISPR-OFFT", "CRISPert", "CrisprBERT", "CnnCrispr", "CRISPR-IP",
]
MODEL_COLORS = {
    "CRISCross_AG": COLORS["jaxgold"],
    "CRISCross_seq": COLORS["seagrey"],
    **dict(zip(NON_CRISCROSS_MODELS, NON_CRISCROSS_PALETTES[COLOR_SCHEME])),
}

MODEL_NAMES = {
    "CRISCross_AG": "CRISCross",
    "CRISCross_seq": "CRISCross",
    "CRISOT": "CRISOT",
    "CRISPR-OFFT": "CRISPR-OFFT",
    "CRISPert": "CRISPert",
    "CrisprBERT": "CrisprBERT",
    "CnnCrispr": "CnnCrispr",
    "CRISPR-IP": "CRISPR-IP",
}

DATASET_ORDER = ["k562_deepcrispr", "k562_full", "ipscs_full"]
DATASET_TITLES = {
    "k562_deepcrispr": "K562 (DeepCRISPR)",
    "ipscs_full": "WTC-11",
    "k562_full": "K562 (CRISPRoffT)",
}


def series_key(row):
    if row["model"] != "CRISCross":
        return row["model"]
    return "CRISCross_AG" if row["features"] == AG_STRING else "CRISCross_seq"


def main():
    df = pd.read_csv("CrossCelltypeThreePanelCombined.tsv", sep="\t")
    df["series"] = df.apply(series_key, axis=1)

    # Mean recall@90%precision per model/dataset.
    bar_df = df[df["series"].isin(SERIES_ORDER)].groupby(
        ["series", "dataset"], as_index=False
    )["recall_at_precision90"].mean()
    xpos_map = {s: i for i, s in enumerate(SERIES_ORDER)}
    bar_df["xpos"] = bar_df["series"].map(xpos_map)

    fig = make_subplots(
        rows=1, cols=len(DATASET_ORDER),
        horizontal_spacing=0.05,
        subplot_titles=[DATASET_TITLES[d] for d in DATASET_ORDER],
    )
    fig.update_annotations(font=dict(size=PT7))

    tick_vals = [xpos_map[s] for s in SERIES_ORDER]
    tick_txt = [MODEL_NAMES[s] for s in SERIES_ORDER]

    added_legends = set()

    for col, ds in enumerate(DATASET_ORDER, start=1):
        suf = "" if col == 1 else str(col)
        sdf = bar_df[bar_df["dataset"] == ds]
        for _, row in sdf.iterrows():
            showlegend = row["series"] not in added_legends
            added_legends.add(row["series"])
            fig.add_trace(
                go.Bar(
                    x=[row["xpos"]],
                    y=[row["recall_at_precision90"]],
                    marker=dict(color=MODEL_COLORS[row["series"]],
                                line=dict(color="black", width=1)),
                    showlegend=showlegend,
                    name=MODEL_NAMES[row["series"]],
                    legendgroup=row["series"],
                    width=0.6,
                ),
                row=1, col=col,
            )
            if pd.notna(row["recall_at_precision90"]):
                fig.add_annotation(
                    x=row["xpos"], y=row["recall_at_precision90"],
                    text=f"{row['recall_at_precision90']:.3f}",
                    showarrow=False,
                    xref=f"x{suf}", yref=f"y{suf}",
                    yanchor="bottom", xanchor="center",
                    textangle=90, font=dict(size=PT7),
                )

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_txt, tickangle=90,
            row=1, col=col,
        )

    fig.update_yaxes(title="Recall@90%Precision", range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1)

    fig.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=320,
        legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
        margin=dict(r=20, b=90, t=30),
    )

    fig.write_html("Figures/CrossCelltypeRecallAt90.html")
    fig.write_image("Figures/CrossCelltypeRecallAt90.svg")
    print("Written: Figures/CrossCelltypeRecallAt90.html/svg")


if __name__ == "__main__":
    main()
