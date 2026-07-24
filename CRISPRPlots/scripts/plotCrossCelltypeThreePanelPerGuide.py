import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, PT7
from plotly.subplots import make_subplots

AG_STRING = "Sequence & AlphaGenome ATAC-seq"

# "warm" uses the warm_axis palette below; "cold" swaps in the cold palette
# for every model other than CRISCross, which keeps its own colors either way.
COLOR_SCHEME = "cold"

# --- Row 1: same as plotCrossCelltypeThreePanel.py's row 1, except
# CRISOT_changeseq is dropped (kept only via CRISOT_tcell, matching row 2)
# and bars use the same per-model warm_axis coloring as row 2 instead of a
# flat AG/seq-only grouping — so there's a single shared legend (below
# row 2) instead of a separate one under row 1.

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
# either scheme; the warm/cold palette (in row-2/SVG legend order) covers
# the other six models. Shared between row 1 and row 2 so both use
# identical bar colors.
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

DATASET_ORDER = ["k562_deepcrispr", "k562_full", "ipscs_full"]
DATASET_TITLES = {
    "k562_deepcrispr": "K562 (DeepCRISPR)",
    "ipscs_full": "WTC-11",
    "k562_full": "K562 (CRISPRoffT)",
}

# --- Row 2: per-guide AUC-PR for the k562_full (CRISPR-OFFT) dataset,
# all 5 guides, all models. CRISOT_changeseq is dropped, keeping only
# CRISOT_tcell (displayed simply as "CRISOT"). Display names and legend
# order are taken from Figures/CrossCellType_ThreePanel_v3.svg.
PERGUIDE_DATASET = "k562_full"

# Kept identical to SERIES_ORDER so row 1 and row 2 bar/legend ordering match.
ROW2_SERIES_ORDER = SERIES_ORDER

ROW2_NAMES = {
    "CRISCross_AG": "CRISCross-Sequence & AG ATAC",
    "CRISCross_seq": "CRISCross-Sequence",
    "CRISOT": "CRISOT",
    "CRISPR-OFFT": "CRISPR-OFFT",
    "CRISPert": "CRISPert",
    "CrisprBERT": "CrisprBERT",
    "CnnCrispr": "CnnCrispr",
    "CRISPR-IP": "CRISPR-IP",
}

N_ROW2_SERIES = len(ROW2_SERIES_ORDER)
ROW2_GROUP_GAP = 1  # empty slots between guide clusters
ROW2_GROUP_WIDTH = N_ROW2_SERIES + ROW2_GROUP_GAP


def series_key(row):
    if row["model"] != "CRISCross":
        return row["model"]
    return "CRISCross_AG" if row["features"] == AG_STRING else "CRISCross_seq"


def main():
    df = pd.read_csv("CrossCelltypeThreePanelCombined.tsv", sep="\t")
    df["series"] = df.apply(series_key, axis=1)

    # Baseline (random-classifier) AUC-PR per dataset for row 1.
    guide_counts = df[df["model"] == "CRISCross"].drop_duplicates(["dataset", "guide_id"])
    baselines = guide_counts.groupby("dataset").apply(
        lambda g: g["n_positives"].sum() / g["n_samples"].sum(), include_groups=False
    ).to_dict()

    # --- Row 1 data: mean AUC-PR per model/dataset ---
    bar_df = df[df["series"].isin(SERIES_ORDER)].groupby(["series", "dataset"], as_index=False)["aucpr"].mean()
    xpos_map = {s: i for i, s in enumerate(SERIES_ORDER)}
    bar_df["xpos"] = bar_df["series"].map(xpos_map)

    # --- Row 2 data: per-guide AUC-PR, k562_full dataset, all 5 guides,
    # ordered by descending n_positives ---
    perguide_df = df[(df["dataset"] == PERGUIDE_DATASET) & (df["series"].isin(ROW2_SERIES_ORDER))].copy()
    guide_order = (
        perguide_df.drop_duplicates("guide_id")
        .sort_values("n_positives", ascending=False)["guide_id"]
        .tolist()
    )
    guide_pos_map = {g: i for i, g in enumerate(guide_order)}
    series_pos_map = {s: i for i, s in enumerate(ROW2_SERIES_ORDER)}
    perguide_df["xpos"] = (
        perguide_df["guide_id"].map(guide_pos_map) * ROW2_GROUP_WIDTH
        + perguide_df["series"].map(series_pos_map)
    )
    n_pos_by_guide = perguide_df.drop_duplicates("guide_id").set_index("guide_id")["n_positives"]

    fig = make_subplots(
        rows=2, cols=len(DATASET_ORDER),
        specs=[[{}, {}, {}], [{"colspan": 3}, None, None]],
        horizontal_spacing=0.05,
        vertical_spacing=0.21,
        row_heights=[0.4, 0.6],
        subplot_titles=[DATASET_TITLES[d] for d in DATASET_ORDER]
        + [f"{DATASET_TITLES[PERGUIDE_DATASET]} - Per Guide"],
    )
    fig.update_annotations(font=dict(size=PT7))

    tick_vals = [xpos_map[s] for s in SERIES_ORDER]
    row1_names = {**ROW2_NAMES, "CRISCross_AG": "CRISCross", "CRISCross_seq": "CRISCross"}
    tick_txt = [row1_names[s] for s in SERIES_ORDER]

    n_cols = len(DATASET_ORDER)

    def axis_suffix(row, col):
        n = (row - 1) * n_cols + col
        return "" if n == 1 else str(n)

    # --- Row 1: bars ---
    for col, ds in enumerate(DATASET_ORDER, start=1):
        suf = axis_suffix(1, col)
        sdf = bar_df[bar_df["dataset"] == ds]
        for _, row in sdf.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["xpos"]],
                    y=[row["aucpr"]],
                    marker=dict(color=MODEL_COLORS[row["series"]],
                                line=dict(color="black", width=1)),
                    showlegend=False,
                    width=0.6,
                ),
                row=1, col=col,
            )
            fig.add_annotation(
                x=row["xpos"], y=row["aucpr"],
                text=f"{row['aucpr']:.3f}",
                showarrow=False,
                xref=f"x{suf}", yref=f"y{suf}",
                yanchor="bottom", xanchor="center",
                textangle=90, font=dict(size=PT7),
            )

        n_series = len(SERIES_ORDER)
        # Reserve one extra x slot past the last bar for the baseline label
        # so its vertical text never overlaps the rightmost bar's own value label.
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_txt, tickangle=90,
            range=[-0.7, n_series + 0.7],
            row=1, col=col,
        )

        baseline_value = baselines[ds]
        fig.add_shape(
            type="line", x0=-0.5, y0=baseline_value, x1=n_series - 0.5, y1=baseline_value,
            xref=f"x{suf}", yref=f"y{suf}",
            line=dict(dash="dash", color="black", width=2),
            row=1, col=col,
        )
        fig.add_annotation(
            text=f"{baseline_value:.3f}", showarrow=False,
            x=n_series, xanchor="center", y=baseline_value, yanchor="bottom",
            xref=f"x{suf}", yref=f"y{suf}", textangle=270,
            font=dict(size=PT7, color="black"),
            row=1, col=col,
        )

    fig.update_yaxes(title="AUC-PR", range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1)
    fig.update_xaxes(tickangle=45, row=1)

    # --- Row 2: per-guide bars, all models ---
    suf2 = axis_suffix(2, 1)
    for series in ROW2_SERIES_ORDER:
        sdf = perguide_df[perguide_df["series"] == series].sort_values("xpos")
        fig.add_trace(
            go.Bar(
                x=sdf["xpos"],
                y=sdf["aucpr"],
                marker=dict(color=MODEL_COLORS[series],
                            line=dict(color="black", width=1)),
                showlegend=True,
                name=ROW2_NAMES[series],
                legendgroup=f"row2_{series}",
                width=0.8,
            ),
            row=2, col=1,
        )
        for _, row in sdf.iterrows():
            fig.add_annotation(
                x=row["xpos"], y=row["aucpr"],
                text=f"{row['aucpr']:.3f}",
                showarrow=False,
                xref=f"x{suf2}", yref=f"y{suf2}",
                yanchor="bottom", xanchor="center",
                textangle=90, font=dict(size=PT7),
            )

    guide_tick_vals = [
        i * ROW2_GROUP_WIDTH + (N_ROW2_SERIES - 1) / 2 for i in range(len(guide_order))
    ]
    guide_tick_txt = [
        f"{g}<br>(n_pos={int(n_pos_by_guide[g])})" for g in guide_order
    ]
    fig.update_xaxes(
        tickvals=guide_tick_vals, ticktext=guide_tick_txt, tickangle=0,
        row=2, col=1,
    )
    fig.update_yaxes(title="AUC-PR", range=[0, 1.2], row=2, col=1)

    fig.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=480,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(r=20, b=50, t=30),
    )

    fig.write_html("Figures/CrossCelltypeThreePanelPerGuide.html")
    fig.write_image("Figures/CrossCelltypeThreePanelPerGuide.svg")
    print("Written: Figures/CrossCelltypeThreePanelPerGuide.html/svg")


if __name__ == "__main__":
    main()
