#!/usr/bin/env python3

import re
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from plotly_template import COLORS, WIDTH

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

INPUT = "SummaryStatsPerWinsize.tsv"

df = pd.read_csv(INPUT, sep="\t")

# ------------------------------------------------------------
# Find raw <-> EX feature pairs
# ------------------------------------------------------------

pattern = re.compile(r"^(EX_)?(.+?)_(mean|median|max|min)_(\d+)$")

results = []

for col in df.columns:

    if not col.startswith("EX_"):
        continue

    raw_col = col[3:]  # remove EX_

    if raw_col not in df.columns:
        continue

    m = pattern.match(col)

    if m is None:
        continue

    _, assay, statistic, window = m.groups()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = df[raw_col].corr(df[col], method="pearson")
    results.append({
        "assay": assay,
        "statistic": statistic,
        "window": int(window),
        "pearson": r
    })

corr_df = pd.DataFrame(results)

# ------------------------------------------------------------
# 2x2 subplot grid: one heatmap per statistic
# ------------------------------------------------------------

statistics_order = ["mean", "median", "min", "max"]

color_scale = [COLORS["white"], COLORS["jaxgold"]]

# Get unique assays in order
assays = sorted(corr_df["assay"].unique())
windows = sorted(corr_df["window"].unique())

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["mean", "median", "min", "max"],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

for idx, stat in enumerate(statistics_order):
    row = idx // 2 + 1
    col = idx % 2 + 1

    sub = corr_df[corr_df["statistic"] == stat]
    heat = sub.pivot(
        index="assay",
        columns="window",
        values="pearson"
    )
    heat = heat.reindex(assays)
    heat = heat.reindex(windows, axis=1)

    # Build text matrix and z-matrix, replace NaN with None for z and "-" for text
    z_matrix = []
    text_matrix = []
    for row_data in heat.values:
        z_row, text_row = [], []
        for val in row_data:
            if pd.isna(val):
                z_row.append(None)
                text_row.append("-")
            else:
                z_row.append(val)
                text_row.append(f"{val:.2f}")
        z_matrix.append(z_row)
        text_matrix.append(text_row)

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=heat.columns.astype(str),
            y=heat.index,
            colorscale=color_scale,
            zmin=0,
            zmax=1,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 8},
            showscale=(row == 2 and col == 2),
            coloraxis="coloraxis",
        ),
        row=row, col=col,
    )

fig.update_layout(
    template="simple_white_custom",
    width=WIDTH,
    height=600,
    coloraxis={"showscale": True, "colorbar":{"len": 0.5}},
    margin=dict(b=40, l=30, t=30),
)

fig.write_html("Figures/PearsonHeatmap.html")
fig.write_image("Figures/PearsonHeatmap.svg", scale=1)
