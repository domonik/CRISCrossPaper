import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_template import WIDTH, PT7, PT6, PT5, PT8

# Multi-panel heatmap view of the precision@R tables built by
# build_precision_tables.py: panel 1 is the cross-dataset summary
# (mean precision@R per model & dataset), panels 2-4 are the per-guide
# precision@R breakdown, one heatmap per dataset. Guide sequences are
# replaced with short per-dataset IDs (G1, G2, ...) on the axis; the
# actual guide sequence and R are kept in the hover text.
#
# Cells are drawn as vector rectangles (fig.add_shape) + text annotations
# rather than go.Heatmap's own fill: Plotly renders a heatmap trace's cell
# colors as a single rasterized PNG embedded in the SVG, so at high zoom it
# looks pixelated. An invisible (opacity=0) go.Heatmap trace is kept per
# panel purely to drive hover tooltips and the shared coloraxis colorbar.

SRC = "CrossCelltypeThreePanelCombined.tsv"
OUT = "Figures/PrecisionAtRHeatmaps"

df = pd.read_csv(SRC, sep="\t")

precision_cols = [c for c in df.columns if c.startswith("precision_at_")]
precision_cols = sorted(precision_cols, key=lambda c: int(c.split("_")[-1]))
MAX_K = max(int(c.split("_")[-1]) for c in precision_cols)

# CRISCross has two feature setups -> treat as two separate "models"
df["model_label"] = df["model"]
mask_criscross = df["model"] == "CRISCross"
df.loc[mask_criscross & (df["features"] == "Sequence & AlphaGenome ATAC-seq"), "model_label"] = "CRISCross (Seq+ATAC)"
df.loc[mask_criscross & (df["features"] == "Sequence only"), "model_label"] = "CRISCross (Seq only)"

other_models = [m for m in df["model"].unique() if m != "CRISCross"]
row_order = ["CRISCross (Seq+ATAC)", "CRISCross (Seq only)"] + list(other_models)
row_order = row_order[::-1]

DATASET_ORDER = ["k562_deepcrispr", "k562_full", "ipscs_full"]
datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
DATASET_LABELS = {
    "ipscs_full": "WTC-11",
    "k562_deepcrispr": "K562 (DeepCrispr)",
    "k562_full": "K562 (CRISPRoffT)",
}
dataset_labels_ordered = [DATASET_LABELS.get(d, d) for d in datasets]


def compute_guide_precision_at_r(ddf):
    # Per guide, R = number of positives for that guide. Look up
    # precision_at_R for each (guide, model); if R exceeds the max recorded
    # k, fall back to precision_at_{MAX_K} and flag the row.
    guides = ddf[["guide_id", "n_positives"]].drop_duplicates()
    rows = []
    for guide_id, r in guides.itertuples(index=False):
        r = int(r)
        flagged = r > MAX_K
        k = min(r, MAX_K)
        col_name = f"precision_at_{k}"
        sub = ddf[ddf["guide_id"] == guide_id]
        for _, row in sub.iterrows():
            rows.append({
                "guide_id": guide_id,
                "n_positives": r,
                "model_label": row["model_label"],
                "value": row[col_name],
                "flagged": flagged,
            })
    return pd.DataFrame(rows)


per_dataset_pr = {}
all_pr = []
for dataset, label in zip(datasets, dataset_labels_ordered):
    ddf = df[df["dataset"] == dataset]
    pr_df = compute_guide_precision_at_r(ddf)
    per_dataset_pr[label] = pr_df
    tagged = pr_df.copy()
    tagged["dataset"] = label
    all_pr.append(tagged)

pr_all = pd.concat(all_pr, ignore_index=True)
any_flagged = bool(pr_all["flagged"].any())

# --- Panel 1: mean precision@R across guides, per model & dataset ---
summary = pr_all.groupby(["model_label", "dataset"])["value"].mean().unstack("dataset")
summary = summary.reindex([m for m in row_order if m in summary.index])
summary = summary[[d for d in dataset_labels_ordered if d in summary.columns]]

# --- Short per-dataset guide IDs, ordered like the Excel tables (ascending R) ---
guide_id_maps = {}
for label, pr_df in per_dataset_pr.items():
    guides_sorted = (
        pr_df[["guide_id", "n_positives"]]
        .drop_duplicates()
        .sort_values(["n_positives", "guide_id"])
        .reset_index(drop=True)
    )
    guide_id_maps[label] = {
        row.guide_id: f"G{i + 1}" for i, row in enumerate(guides_sorted.itertuples(index=False))
    }

# WTC-11 has only a single guide RNA, so its per-guide breakdown (panel b)
# would just repeat the one data point already shown in panel a -- drop it
# from the bottom row while keeping it in the panel a summary.
bottom_dataset_labels = [label for label in dataset_labels_ordered if label != "WTC-11"]

n_guides = {label: len(guide_id_maps[label]) for label in bottom_dataset_labels}
col_widths_raw = [n_guides[label] for label in bottom_dataset_labels]
total_width = sum(col_widths_raw)
column_widths = [w / total_width for w in col_widths_raw]


# --- Color scale evaluated in Python for the vector cell fills ---
# Either a list of hex colors (custom, evenly spaced) or the name of a
# Plotly built-in colorscale (e.g. "Viridis", "RdBu"; see
# plotly.colors.PLOTLY_SCALES for all names).
PALETTE = ['#EDEEEE','#DAE0E0','#C8D1D3','#BAC3C5','#ACB4B7',
       '#94A8AE','#6A9FA9','#3A94A4','#17879A']
# Set to False to draw panel 1 (the cross-dataset summary table) as plain
# white cells instead of the colorscale; panels 2-4 are unaffected.
COLOR_SUMMARY_PANEL = True
# "value": color each cell by its raw precision@R value (shared 0-1 scale
# across all panels).
# "rank": color each cell by its rank *within its own column* (i.e. within a
# dataset/guide), rank 1 (best model) getting the darkest color and the
# worst-performing model the lightest -- regardless of how close/far apart
# the actual values are.
# "relative": color each cell on a 0-1 scale *within its own column*, where 0
# is zero and 1 is that column's best-performing model's value -- other
# models are shaded in proportion to how close they get to the best one.
# Cell text always shows the raw value, regardless of mode.
COLOR_MODE = "rank"  # "value", "rank", or "relative"
_COLORSCALE = pcolors.get_colorscale(PALETTE) if isinstance(PALETTE, str) else [
    [i / (len(PALETTE) - 1), c] for i, c in enumerate(PALETTE)
]

def value_to_rgb(val, vmin=0.0, vmax=1.0):
    t = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    r, g, b = pcolors.sample_colorscale(_COLORSCALE, [t], colortype="tuple")[0]
    return r * 255, g * 255, b * 255


def value_to_color(val, vmin=0.0, vmax=1.0):
    r, g, b = value_to_rgb(val, vmin, vmax)
    return f"rgb({r:.0f},{g:.0f},{b:.0f})"


def _wcag_relative_luminance(r, g, b):
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def readable_text_color(val, vmin=0.0, vmax=1.0):
    # Pick whichever of black/white actually gives the higher WCAG contrast
    # ratio against this exact cell color -- a fixed luminance threshold
    # leaves mid-tone cells (common in this palette) low-contrast either way.
    r, g, b = value_to_rgb(val, vmin, vmax)
    l = _wcag_relative_luminance(r, g, b)
    contrast_white = 1.05 / (l + 0.05)
    contrast_black = (l + 0.05) / 0.05
    return "black" if contrast_black >= contrast_white else "white"


def add_table_panel(fig, row, col, values, row_labels, col_labels, customdata, hovertemplate,
                     flagged=None, text_size=PT6, colored=True):
    """Draw one panel as vector rect cells + text annotations, bolding the best
    (highest) value in each column. An invisible go.Heatmap trace underneath
    supplies hover tooltips and feeds the shared coloraxis colorbar."""
    n_rows, n_cols = values.shape
    best_row_per_col = values.argmax(axis=0)

    if COLOR_MODE == "rank" and n_rows > 1:
        # Rank 1 = best (highest value) within its column. Map rank -> [0, 1]
        # so the darkest palette color goes to rank 1, independent of the
        # column's actual value spread.
        ranks = (-values).argsort(axis=0).argsort(axis=0) + 1
        color_source = (n_rows - ranks) / (n_rows - 1)
    elif COLOR_MODE == "relative":
        # Scale each column by its own best (max) value, so the best model
        # in that column always reaches the darkest color and others are
        # shaded proportionally between 0 and that best value.
        col_max = values.max(axis=0, keepdims=True)
        col_max = np.where(col_max == 0, 1, col_max)
        color_source = values / col_max
    else:
        color_source = values

    fig.add_trace(
        go.Heatmap(
            z=values,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            coloraxis="coloraxis",
            opacity=0,
            customdata=customdata,
            hovertemplate=hovertemplate,
        ),
        row=row, col=col,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            color_val = color_source[i, j]
            fig.add_shape(
                type="rect",
                x0=j, x1=j + 1, y0=i, y1=i + 1,
                line=dict(color="black", width=1),
                fillcolor=value_to_color(color_val) if colored else "white",
                layer="below",
                row=row, col=col,
            )
            txt = f"{val:.3f}"
            if flagged is not None and flagged[i, j]:
                txt += "*"
            if i == best_row_per_col[j]:
                txt = f"<b>{txt}</b>"
            fig.add_annotation(
                text=txt, showarrow=False,
                x=j + 0.5, y=i + 0.5,
                font=dict(size=text_size, color=readable_text_color(color_val) if colored else "black"),
                row=row, col=col,
            )

    fig.update_xaxes(
        range=[0, n_cols], tickvals=[c + 0.5 for c in range(n_cols)], ticktext=col_labels,
        row=row, col=col,
    )
    fig.update_yaxes(
        range=[0, n_rows], tickvals=[r + 0.5 for r in range(n_rows)], ticktext=row_labels,
        row=row, col=col,
    )


HORIZONTAL_SPACING = 0.03

N_BOTTOM_COLS = len(bottom_dataset_labels)

fig = make_subplots(
    rows=2, cols=N_BOTTOM_COLS,
    specs=[[{"colspan": N_BOTTOM_COLS}] + [None] * (N_BOTTOM_COLS - 1), [{}] * N_BOTTOM_COLS],
    column_widths=column_widths,
    row_heights=[0.35, 0.65],
    vertical_spacing=0.13,
    horizontal_spacing=HORIZONTAL_SPACING,
    shared_yaxes=True,
)
# Scientific-figure style panel tags instead of descriptive titles (the
# explanation belongs in the figure caption, not on the figure itself).
fig.add_annotation(
    text="<b>a</b>",
    showarrow=False,
    x=-0.2, xref="paper", xanchor="left",
    y=1.0, yref="paper", yanchor="bottom", yshift=26,
    font=dict(size=PT8, family="Arial"),
)
fig.add_annotation(
    text="<b>b</b>",
    showarrow=False,
    x=-0.2, xref="paper", xanchor="left",
    y=0.62, yref="paper", yanchor="bottom",
    font=dict(size=PT8, family="Arial"),
)

# Per-dataset column headers above panel b, sized to match panel a's dataset
# headers (PT7), computed from the same column_widths/horizontal_spacing
# make_subplots uses so each label centers over its own panel.
_avail = 1 - HORIZONTAL_SPACING * (len(column_widths) - 1)
_cursor = 0.0
for _w, _label in zip(column_widths, bottom_dataset_labels):
    _width = _w * _avail
    fig.add_annotation(
        text=_label, showarrow=False,
        x=_cursor + _width / 2, xref="paper", xanchor="center",
        y=0.62, yref="paper", yanchor="bottom",
        font=dict(size=PT7, family="Arial"),
    )
    _cursor += _width + HORIZONTAL_SPACING

# --- Panel 1 ---
summary_row_labels = summary.index.tolist()
summary_col_labels = summary.columns.tolist()
summary_customdata = [
    [[model, ds] for ds in summary_col_labels] for model in summary_row_labels
]
add_table_panel(
    fig, row=1, col=1,
    values=summary.values,
    row_labels=summary_row_labels, col_labels=summary_col_labels,
    customdata=summary_customdata,
    hovertemplate="Model: %{customdata[0]}<br>Dataset: %{customdata[1]}<br>Precision@R: %{z:.3f}<extra></extra>",
    text_size=PT6,
    colored=COLOR_SUMMARY_PANEL,
)

# --- Panels 2-4: per-guide heatmaps, one per dataset (excludes WTC-11) ---
for idx, label in enumerate(bottom_dataset_labels):
    pr_df = per_dataset_pr[label].copy()
    id_map = guide_id_maps[label]
    pr_df["short_id"] = pr_df["guide_id"].map(id_map)
    inv_id_map = {v: k for k, v in id_map.items()}

    pivot_val = pr_df.pivot(index="model_label", columns="short_id", values="value")
    pivot_val = pivot_val.reindex([m for m in row_order if m in pivot_val.index])
    ordered_cols = sorted(pivot_val.columns, key=lambda s: int(s[1:]))
    pivot_val = pivot_val[ordered_cols]

    pivot_flag = pr_df.pivot(index="model_label", columns="short_id", values="flagged")
    pivot_flag = pivot_flag.reindex(pivot_val.index)[ordered_cols]

    r_lookup = pr_df.drop_duplicates("short_id").set_index("short_id")["n_positives"]

    row_labels = pivot_val.index.tolist()
    customdata = [
        [[model, inv_id_map[col], int(r_lookup[col])] for col in ordered_cols]
        for model in row_labels
    ]

    # Show the guide's number of positives in brackets, on its own line, next to its short ID.
    tick_text = [f"{col}<br>[{int(r_lookup[col])}]" for col in ordered_cols]

    add_table_panel(
        fig, row=2, col=idx + 1,
        values=pivot_val.values,
        row_labels=row_labels, col_labels=tick_text,
        customdata=customdata,
        hovertemplate=(
            "Model: %{customdata[0]}<br>Guide: %{customdata[1]} (R=%{customdata[2]})"
            "<br>Precision@R: %{z:.3f}<extra></extra>"
        ),
        flagged=pivot_flag.values,
        text_size=PT5,
    )

fig.update_layout(
    template="simple_white_custom",
    width=WIDTH,
    height=500,
    coloraxis=dict(
        colorscale=_COLORSCALE,
        cmin=0, cmax=1,
        # The colorbar reflects the raw 0-1 precision scale, which only
        # matches the cell fill colors in COLOR_MODE == "value"; in "rank"
        # and "relative" modes fills are computed per-column, so the shared
        # bar would be misleading and is hidden instead.
        colorbar=dict(title=dict(text="Precision@R"), len=0.6) if COLOR_MODE == "value" else None,
        showscale=COLOR_MODE == "value",
    ),
    margin=dict(l=110, r=50, t=65, b=30),
)
# Table-like axes: no tick marks, horizontal column headers on top, solid black border on all sides.
fig.update_xaxes(
    ticks="", side="top", tickangle=0,
    showline=True, linecolor="black", linewidth=1, mirror=True,
    showgrid=False, zeroline=False,
)
fig.update_yaxes(
    ticks="", tickfont=dict(size=PT6),
    showline=True, linecolor="black", linewidth=1, mirror=True,
    showgrid=False, zeroline=False,
)
# Dataset names on panel 1's header row get a bit more emphasis than the rest.
fig.update_xaxes(tickfont=dict(size=PT7), row=1, col=1)

fig.write_html(f"{OUT}.html")
fig.write_image(f"{OUT}.svg")
print(f"Written: {OUT}.html/svg")
