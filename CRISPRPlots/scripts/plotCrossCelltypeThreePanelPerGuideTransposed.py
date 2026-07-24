import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, PT7, PT6
from plotly.subplots import make_subplots

AG_STRING = "Sequence & AlphaGenome ATAC-seq"

# Transposed layout of plotCrossCelltypeThreePanelPerGuide.py:
# - column 1 holds the three per-dataset panels that used to be row 1
#   (stacked instead of side-by-side), each with model/dataset axes swapped
#   to horizontal bars.
# - column 2 holds the per-guide panel that used to be row 2, spanning all
#   three rows, also as horizontal bars.
# Per-model colors are dropped: only the two CRISCross variants keep their
# own colors (jaxgold for +AlphaGenome ATAC, seagrey for sequence-only);
# every other model shares a single neutral grey. Model identity is instead
# carried by text (axis tick labels in column 1, bar labels in column 2) on
# every subplot.

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

OTHER_MODEL_COLOR = "#CDD2D4"

MODEL_COLORS = {s: OTHER_MODEL_COLOR for s in SERIES_ORDER}
MODEL_COLORS["CRISCross_AG"] = COLORS["jaxgold"]
MODEL_COLORS["CRISCross_seq"] = COLORS["seagrey"]

MODEL_LABELS = {
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
    "ipscs_full": "IPSC",
    "k562_full": "K562 (CRISPRoffT)",
}

PERGUIDE_DATASET = "k562_full"
ROW2_SERIES_ORDER = SERIES_ORDER
N_ROW2_SERIES = len(ROW2_SERIES_ORDER)
ROW2_GROUP_GAP = 1  # empty slots between guide clusters
ROW2_GROUP_WIDTH = N_ROW2_SERIES + ROW2_GROUP_GAP

# Axis numbering for our fixed 3x2 grid (row-major, skipping the cells
# covered by column 2's rowspan). Plotly assigns axis suffixes in this order.
AXIS_SUFFIX = {(1, 1): "", (1, 2): "2", (2, 1): "3", (3, 1): "4"}


def series_key(row):
    if row["model"] != "CRISCross":
        return row["model"]
    return "CRISCross_AG" if row["features"] == AG_STRING else "CRISCross_seq"


def main():
    df = pd.read_csv("CrossCelltypeThreePanelCombined.tsv", sep="\t")
    df["series"] = df.apply(series_key, axis=1)

    # Baseline (random-classifier) AUC-PR per dataset for column 1.
    guide_counts = df[df["model"] == "CRISCross"].drop_duplicates(["dataset", "guide_id"])
    baselines = guide_counts.groupby("dataset").apply(
        lambda g: g["n_positives"].sum() / g["n_samples"].sum(), include_groups=False
    ).to_dict()

    # --- Column 1 data: mean AUC-PR per model/dataset ---
    bar_df = df[df["series"].isin(SERIES_ORDER)].groupby(["series", "dataset"], as_index=False)["aucpr"].mean()
    ypos_map = {s: i for i, s in enumerate(SERIES_ORDER)}
    bar_df["ypos"] = bar_df["series"].map(ypos_map)

    # --- Column 2 data: per-guide AUC-PR, k562_full dataset, all guides ---
    perguide_df = df[(df["dataset"] == PERGUIDE_DATASET) & (df["series"].isin(ROW2_SERIES_ORDER))].copy()
    guide_order = (
        perguide_df.drop_duplicates("guide_id")
        .sort_values("n_positives", ascending=False)["guide_id"]
        .tolist()
    )
    guide_pos_map = {g: i for i, g in enumerate(guide_order)}
    series_pos_map = {s: i for i, s in enumerate(ROW2_SERIES_ORDER)}
    perguide_df["ypos"] = (
        perguide_df["guide_id"].map(guide_pos_map) * ROW2_GROUP_WIDTH
        + perguide_df["series"].map(series_pos_map)
    )
    n_series = len(SERIES_ORDER)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{}, {"rowspan": 3}],
            [{}, None],
            [{}, None],
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.07,
        column_widths=[0.4, 0.6],
        subplot_titles=[
            DATASET_TITLES[DATASET_ORDER[0]],
            f"{DATASET_TITLES[PERGUIDE_DATASET]} - Per Guide",
            DATASET_TITLES[DATASET_ORDER[1]],
            DATASET_TITLES[DATASET_ORDER[2]],
        ],
    )
    fig.update_annotations(font=dict(size=PT7))

    tick_vals = [ypos_map[s] for s in SERIES_ORDER]
    tick_txt = [MODEL_LABELS[s] for s in SERIES_ORDER]

    # --- Column 1: three stacked per-dataset panels, horizontal bars ---
    for i, ds in enumerate(DATASET_ORDER):
        row = i + 1
        suf = AXIS_SUFFIX[(row, 1)]
        sdf = bar_df[bar_df["dataset"] == ds]
        for _, r in sdf.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[r["aucpr"]],
                    y=[r["ypos"]],
                    orientation="h",
                    marker=dict(color=MODEL_COLORS[r["series"]], line=dict(color="black", width=1)),
                    text=[f"{r['aucpr']:.3f}"],
                    textposition="outside",
                    textfont=dict(size=PT7),
                    showlegend=False,
                    width=0.6,
                ),
                row=row, col=1,
            )

        fig.update_yaxes(
            tickvals=tick_vals, ticktext=tick_txt,
            range=[n_series - 0.3, -0.7],
            row=row, col=1,
        )
        fig.update_xaxes(range=[0, 1.3], row=row, col=1)

        baseline_value = baselines[ds]
        fig.add_shape(
            type="line", x0=baseline_value, y0=-0.5, x1=baseline_value, y1=n_series - 0.5,
            xref=f"x{suf}", yref=f"y{suf}",
            line=dict(dash="dash", color="black", width=2),
            row=row, col=1,
        )
        fig.add_annotation(
            text=f"{baseline_value:.3f}", showarrow=False,
            x=baseline_value, xanchor="left", y=-0.5, yanchor="bottom",
            xref=f"x{suf}", yref=f"y{suf}",
            font=dict(size=PT7, color="black"),
        )

    fig.update_xaxes(title="AUC-PR", row=3, col=1)

    # --- Column 2: per-guide panel, all models, horizontal grouped bars ---
    suf2 = AXIS_SUFFIX[(1, 2)]
    for series in ROW2_SERIES_ORDER:
        sdf = perguide_df[perguide_df["series"] == series].sort_values("ypos")
        fig.add_trace(
            go.Bar(
                x=sdf["aucpr"],
                y=sdf["ypos"],
                orientation="h",
                marker=dict(color=MODEL_COLORS[series], line=dict(color="black", width=1)),
                text=[f"{v:.3f}" for v in sdf["aucpr"]],
                textposition="outside",
                textfont=dict(size=PT6),
                showlegend=False,
                width=0.8,
            ),
            row=1, col=2,
        )

    max_ypos = (len(guide_order) - 1) * ROW2_GROUP_WIDTH + (N_ROW2_SERIES - 1)
    row2_ticks = perguide_df.sort_values("ypos")
    fig.update_yaxes(
        tickvals=row2_ticks["ypos"].tolist(),
        ticktext=[MODEL_LABELS[s] for s in row2_ticks["series"]],
        range=[max_ypos + 0.5, -0.5],
        row=1, col=2,
    )
    fig.update_xaxes(title="AUC-PR", range=[0, 1.3], row=1, col=2)

    # --- Legend: static entries explaining the color coding ---
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker=dict(color=COLORS["jaxgold"], line=dict(color="black", width=1)),
        name="Sequence & AlphaGenome ATAC-seq", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker=dict(color=COLORS["seagrey"], line=dict(color="black", width=1)),
        name="Sequence only", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker=dict(color=OTHER_MODEL_COLOR, line=dict(color="black", width=1)),
        name="Other models", showlegend=True,
    ))

    fig.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=650,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(r=20, b=50, t=30, l=60),
    )

    fig.write_html("Figures/CrossCelltypeThreePanelPerGuideTransposed.html")
    fig.write_image("Figures/CrossCelltypeThreePanelPerGuideTransposed.svg")
    print("Written: Figures/CrossCelltypeThreePanelPerGuideTransposed.html/svg")


if __name__ == "__main__":
    main()
