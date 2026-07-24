import math

import pandas as pd
import plotly.graph_objects as go
from plotly_template import WIDTH, COLORS, PT7
from plotly.subplots import make_subplots

AG_STRING = "Sequence & AlphaGenome ATAC-seq"
SEQ_STRING = "Sequence only"

# Order in which models/series appear along the bar-chart x-axis and in legends.
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

# Display names used for the bar-chart x-axis tick labels (row 1). Both
# CRISCross bars are labeled the same way; the legend disambiguates them
# via the feature-group color (AG vs seq-only).
SERIES_NAMES = {
    "CRISCross_AG": "CRISCross",
    "CRISCross_seq": "CRISCross",
    "CRISPR-OFFT": "CRISPR-OFFT",
    "CRISPert": "CRISPert",
    "CnnCrispr": "CnnCrispr",
    "CRISPR-IP": "CRISPR-IP",
    "CRISOT": "CRISOT",
    "CrisprBERT": "CrisprBERT",
}

# Display names used for the precision@K line-panel legend (row 2), where
# the two CRISCross series need distinct labels since they're separate lines.
LINE_SERIES_NAMES = {
    **SERIES_NAMES,
    "CRISCross_AG": "CRISCross + ATAC-seq",
    "CRISCross_seq": "CRISCross (seq only)",
}

# Bar colors are keyed off feature group only (AG vs seq-only), matching the
# original two-panel plot.
BAR_COLORS = {
    "CRISCross_AG": COLORS["jaxgold"],
    "CRISCross_seq": COLORS["seagrey"],
    "CRISPR-OFFT": COLORS["seagrey"],
    "CRISPert": COLORS["seagrey"],
    "CnnCrispr": COLORS["seagrey"],
    "CRISPR-IP": COLORS["seagrey"],
    "CRISOT": COLORS["seagrey"],
    "CrisprBERT": COLORS["seagrey"],
}
BAR_LEGEND_GROUP = {
    "CRISCross_AG": AG_STRING,
    "CRISCross_seq": SEQ_STRING,
    "CRISPR-OFFT": SEQ_STRING,
    "CRISPert": SEQ_STRING,
    "CnnCrispr": SEQ_STRING,
    "CRISPR-IP": SEQ_STRING,
    "CRISOT": SEQ_STRING,
    "CrisprBERT": SEQ_STRING,
}

# Each series gets its own distinct color in the precision@K line panels
# (reusing the palette from plotPrecisionVsKTwoPanel.py for the old models).
LINE_COLORS = {
    "CRISCross_AG": COLORS["jaxgold"],
    "CRISCross_seq": COLORS["jaxpetrol"],
    "CRISPR-OFFT": COLORS["seagrey"],
    "CRISPert": COLORS["cat9"],
    "CnnCrispr": COLORS["cat3"],
    "CRISPR-IP": COLORS["cat4"],
    "CRISOT": COLORS["cat6"],
    "CrisprBERT": COLORS["cat8"],
}

DATASET_ORDER = ["k562_deepcrispr", "ipscs_full", "k562_full"]
DATASET_TITLES = {
    "k562_deepcrispr": "K562 (DeepCRISPR)",
    "ipscs_full": "IPSC",
    "k562_full": "K562 (Full / IVK562)",
}

K_VALUES = [5, 10, 25, 50, 100, 500, 1000]
PRECISION_COLS = [f"precision_at_{k}" for k in K_VALUES]

# Distinct marker symbol per series, so overlapping precision@K lines stay
# distinguishable even when colors are close.
MARKER_SYMBOLS = {
    "CRISCross_AG": "circle",
    "CRISCross_seq": "diamond",
    "CRISPR-OFFT": "square",
    "CRISPert": "triangle-up",
    "CnnCrispr": "triangle-down",
    "CRISPR-IP": "cross",
    "CRISOT": "star",
    "CrisprBERT": "pentagon",
}


def series_key(row):
    if row["model"] != "CRISCross":
        return row["model"]
    return "CRISCross_AG" if row["features"] == AG_STRING else "CRISCross_seq"


def main():
    df = pd.read_csv("CrossCelltypeThreePanelCombined.tsv", sep="\t")
    df["series"] = df.apply(series_key, axis=1)

    # Baseline (random-classifier) AUC-PR per dataset: the overall positive
    # rate (total positives / total samples) across the guides that make up
    # that dataset's panel. Computed from the CRISCross rows since those
    # carry the n_positives/n_samples derived from the raw guide tables, one
    # row per unique guide (features doesn't affect guide-level counts).
    guide_counts = df[df["model"] == "CRISCross"].drop_duplicates(["dataset", "guide_id"])
    baselines = guide_counts.groupby("dataset").apply(
        lambda g: g["n_positives"].sum() / g["n_samples"].sum(), include_groups=False
    ).to_dict()

    # --- Panel row 1: mean AUC-PR per model/dataset ---
    bar_df = df.groupby(["series", "dataset"], as_index=False)["aucpr"].mean()
    xpos_map = {s: i for i, s in enumerate(SERIES_ORDER)}
    bar_df["xpos"] = bar_df["series"].map(xpos_map)

    # --- Panel row 2: mean precision@K per model/dataset ---
    line_df = df.groupby(["series", "dataset"], as_index=False)[PRECISION_COLS].mean()
    line_df["model_name"] = line_df["series"].map(LINE_SERIES_NAMES)

    out_cols = ["series", "model_name", "dataset"] + PRECISION_COLS
    out_file = "Tables/CrossCelltypeThreePanel_PrecisionAtK.tsv"
    line_df[out_cols].sort_values(["dataset", "series"]).to_csv(out_file, sep="\t", index=False)
    print(f"Written: {out_file}")

    # --- Panel row 3: mean recall@90%precision per model/dataset ---
    recall_df = df.groupby(["series", "dataset"], as_index=False)["recall_at_precision90"].mean()
    recall_df["xpos"] = recall_df["series"].map(xpos_map)

    fig = make_subplots(
        rows=3, cols=len(DATASET_ORDER),
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
        shared_yaxes="rows",
        subplot_titles=[DATASET_TITLES[d] for d in DATASET_ORDER] + [""] * 2 * len(DATASET_ORDER),
    )
    fig.update_annotations(font=dict(size=PT7))

    tick_vals = [xpos_map[s] for s in SERIES_ORDER]
    tick_txt = [SERIES_NAMES[s] for s in SERIES_ORDER]

    added_bar_legends = set()
    added_line_legends = set()

    n_cols = len(DATASET_ORDER)

    def axis_suffix(row, col):
        n = (row - 1) * n_cols + col
        return "" if n == 1 else str(n)

    for col, ds in enumerate(DATASET_ORDER, start=1):
        suf = axis_suffix(1, col)
        suf3 = axis_suffix(3, col)
        # --- Row 1: bars ---
        sdf = bar_df[bar_df["dataset"] == ds]
        for _, row in sdf.iterrows():
            group = BAR_LEGEND_GROUP[row["series"]]
            showlegend = group not in added_bar_legends
            added_bar_legends.add(group)
            fig.add_trace(
                go.Bar(
                    x=[row["xpos"]],
                    y=[row["aucpr"]],
                    marker=dict(color=BAR_COLORS[row["series"]],
                                line=dict(color="black", width=1)),
                    showlegend=showlegend,
                    name=group,
                    legendgroup=group,
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

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_txt, tickangle=90,
            row=1, col=col,
        )

        baseline_value = baselines[ds]
        fig.add_shape(
            type="line", x0=0, y0=baseline_value, x1=1, y1=baseline_value,
            xref=f"x{suf} domain", yref=f"y{suf}",
            line=dict(dash="dash", color="black", width=2),
            row=1, col=col,
        )
        fig.add_annotation(
            text=f"{baseline_value:.3f}", showarrow=False,
            x=1, xanchor="right", y=baseline_value, yanchor="bottom",
            xref=f"x{suf} domain", yref=f"y{suf}", textangle=270,
            font=dict(size=PT7, color="black"),
            row=1, col=col,
        )

        # --- Row 2: precision@K lines ---
        ldf = line_df[line_df["dataset"] == ds]
        for series in SERIES_ORDER:
            mrow = ldf[ldf["series"] == series]
            if mrow.empty:
                continue
            mrow = mrow.iloc[0]
            y = [mrow[f"precision_at_{k}"] for k in K_VALUES]
            name = LINE_SERIES_NAMES[series]
            showlegend = series not in added_line_legends
            added_line_legends.add(series)
            fig.add_trace(
                go.Scatter(
                    x=K_VALUES, y=y,
                    mode="lines+markers",
                    name=name,
                    legendgroup=f"line_{series}",
                    line=dict(color=LINE_COLORS[series], width=1.2),
                    marker=dict(size=3.5, symbol=MARKER_SYMBOLS[series]),
                    showlegend=showlegend,
                ),
                row=2, col=col,
            )

        fig.update_xaxes(
            type="log",
            range=[math.log10(5), math.log10(1000)],
            tickvals=K_VALUES,
            ticktext=[str(k) for k in K_VALUES],
            title_text="K",
            row=2, col=col,
        )

        # --- Row 3: recall@90%precision bars ---
        rdf = recall_df[recall_df["dataset"] == ds]
        for _, row in rdf.iterrows():
            group = BAR_LEGEND_GROUP[row["series"]]
            showlegend = group not in added_bar_legends
            added_bar_legends.add(group)
            fig.add_trace(
                go.Bar(
                    x=[row["xpos"]],
                    y=[row["recall_at_precision90"]],
                    marker=dict(color=BAR_COLORS[row["series"]],
                                line=dict(color="black", width=1)),
                    showlegend=showlegend,
                    name=group,
                    legendgroup=group,
                    width=0.6,
                ),
                row=3, col=col,
            )
            if pd.notna(row["recall_at_precision90"]):
                fig.add_annotation(
                    x=row["xpos"], y=row["recall_at_precision90"],
                    text=f"{row['recall_at_precision90']:.3f}",
                    showarrow=False,
                    xref=f"x{suf3}", yref=f"y{suf3}",
                    yanchor="bottom", xanchor="center",
                    textangle=90, font=dict(size=PT7),
                )

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_txt, tickangle=90,
            row=3, col=col,
        )

    fig.update_yaxes(title="AUC-PR", range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(title="Precision@K", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title="Recall@90%Precision", range=[0, 1.15], row=3, col=1)

    fig.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=620,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(r=20, b=10),
    )

    fig.write_html("Figures/CrossCelltypeThreePanel.html")
    fig.write_image("Figures/CrossCelltypeThreePanel.svg")
    print("Written: Figures/CrossCelltypeThreePanel.html/svg")


if __name__ == "__main__":
    main()
