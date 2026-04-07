import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gatherDistributions import FEATURES
from plotly_template import PT7, PT6, COLORS, WIDTH
import numpy as np
import pandas as pd


def plot_median_quantiles(prefix, outfile_postfix,  colors=["#1F77B4", "#FF7F0E"], opacity=0.4):
    """
    values: shape (10, N)
    Plot:
        - median (solid)
        - q25–q75 (filled band)
        - mean (dashed)
    """
    hs = 0.07
    feats = [f for f in FEATURES if f.startswith("EX")]
    fig = make_subplots(
        rows=len(feats),
          cols=4, 
          x_title="Distance [nt]",
        #row_titles=[f[3:] for f in feats],
        horizontal_spacing=hs
    )
    fig.update_annotations(font=dict(size=PT6))
    for anno in fig.layout.annotations:
        anno.update(y=anno.y+0.06)
    for i, m_feat in enumerate(feats):
        fig.add_annotation(
                text=m_feat[3:],
                x=1,
                y=0.5,
                yanchor="middle",
                xanchor="left",
                showarrow=False,
                xref="paper",
                yref="y domain" if i == 0 else f"y{i*4 + 1} domain",
                textangle=90,
                font=dict(size=PT7)

            )
        for k, feature in enumerate([m_feat, m_feat[3:]]):
            values = np.load(f"{prefix}{feature}_{outfile_postfix}")
            x_v = values.shape[1]//2
            x = np.arange(-x_v, x_v)


            if feature.startswith("H3"):
                x = x * 128

            # Positive
            pos_mean   = values[0]
            pos_median = values[1]
            pos_q25    = values[3]
            pos_q75    = values[4]

            # Negative
            neg_mean   = values[5]
            neg_median = values[6]
            neg_q25    = values[8]
            neg_q75    = values[9]


            # --- Positive IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([pos_q25, pos_q75[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Positive IQR",
                fillcolor=colors[0],
                line=dict(width=0),

            ), col=k*2+2, row=i+1)

            # Positive median
            fig.add_trace(go.Scatter(
                x=x, y=pos_median,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0], dash="dot",)
            ), col=k*2+2, row=i+1)

            # Positive mean (dashed)
            fig.add_trace(go.Scatter(
                x=x, y=pos_mean,
                mode="lines",
                name="Positive Mean",
                line=dict(width=1, color=colors[0])
            ), col=k*2+1, row=i+1)

            # --- Negative IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([neg_q25, neg_q75[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Negative IQR",
                fillcolor=colors[1],
                line=dict(width=0)
            ), col=k*2+2, row=i+1)

            # Negative median
            fig.add_trace(go.Scatter(
                x=x, y=neg_median,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1], dash="dot",)
            ), col=k*2+2, row=i+1)

            # Negative mean (dashed)
            fig.add_trace(go.Scatter(
                x=x, y=neg_mean,
                mode="lines",
                name="Negative Mean",
                line=dict(width=1, color=colors[1])
            ), col=k*2+1, row=i+1)

    fig.update_traces(showlegend=False)

    fig.update_xaxes(dtick=4000)
    fig.update_yaxes(
        col=1,
        title=dict(text="normalized signal")
    )
    fig.update_yaxes(
        col=3,
        title=dict(text="AlphaGenome score [AU]")
    )
    fig.add_traces(
        [
            go.Bar(
                x=[None],
                y=[None],
                name="GUIDE-seq",
                marker=dict(color=colors[0]),


            ),
            go.Bar(
                x=[None],
                y=[None],
                name="Cas-OFFfinder",
                marker=dict(color=colors[1]),


            ),
            go.Scatter(
                x=[None],
                y=[None],
                line=dict(width=1, dash="dot", color="black"),
                name="Median",
                marker=dict(size=0),
                mode="lines",



            ),
            go.Scatter(
                x=[None],
                y=[None],
                line=dict(width=1, color="black"),
                marker=dict(size=0),
                mode="lines",

                name="Mean"

            ),
            go.Scatter(
                x=[None],
                y=[None],
                fill="toself",
                opacity=opacity,
                name="IQR",
                fillcolor="black",
                line=dict(width=0),
                marker=dict(size=0),
                mode="lines",

            )
        ]
    )
    y = 1.05
    fig.add_shape(
        type="line",
        x0=0.05, x1=0.5-0.5*hs-0.05 ,          # full width of the row
        y0=y,
        y1=y,
        xref="paper",        # span across subplot domains
        yref="paper",           # reference the y-axis of the first subplot in that row
        line=dict(color="black", width=1)
    )
    fig.add_shape(
        type="line",
        x0=0.5+0.5*hs + 0.05, x1=1-0.05,          # full width of the row
        y0=y,
        y1=y,
        xref="paper",        # span across subplot domains
        yref="paper",           # reference the y-axis of the first subplot in that row
        line=dict(color="black", width=1)
    )
    fig.add_annotation(
        text="Experimental",
        x=(0.05 +0.5-0.5*hs - 0.05) / 2,
        y=y,
        yanchor="bottom",
        xanchor="center",
        showarrow=False,
        xref="paper",
        yref="paper"
    )

    fig.add_annotation(
        text="AlphaGenome",
        x=(0.5+0.5*hs + 0.05 + 1-0.05) / 2,
        y=y,
        yanchor="bottom",
        xanchor="center",
        showarrow=False,
        xref="paper",
        yref="paper"

    )


    fig.update_layout(
        template="simple_white_custom",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, yref="paper"),
        margin=dict(t=10, b=30),
        height=350,
        width=(WIDTH // 4) * 3
    )

    return fig


def plot_results(prefix, outfile):
    #fig = plot_mean_std(values)
    fig = plot_median_quantiles(outfile_postfix=outfile, prefix=prefix, colors=[COLORS["jaxpetrol"], COLORS["jaxgold"]])
    fig.write_image("Figures/DistributionRes.svg")
    fig.write_html("Figures/DistributionRes.html")



if __name__ == "__main__":
    plot_results(outfile="SummaryNumpyArray.npy", prefix="GenomicSummary/")