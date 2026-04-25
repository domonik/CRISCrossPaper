import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gatherDistributions import FEATURES
from plotly_template import PT7, PT6, COLORS, WIDTH
import numpy as np
import pandas as pd



def plot_mean_std(prefix, outfile_postfix,  colors=["#1F77B4", "#FF7F0E"], opacity=0.4):
    """
    values: shape (10, N)
    Plot:
        - median (solid)
        - q25–q75 (filled band)
        - mean (dashed)
    """
    hs = 0.05
    extra = 0.02
    feats = [f for f in FEATURES if f.startswith("EX")]
    specs = [
        [
            {"l": extra},                      # col 1 - normal
            {"r": extra},            # col 2 - push right neighbour away
            {"l": extra},                      # col 3 - normal
            {"r": extra}                       # col 4 - normal
        ]
        for _ in range(len(feats))
    ]
    fig = make_subplots(
        rows=len(feats),
          cols=4, 
          x_title="Distance [nt]",
        #row_titles=[f[3:] for f in feats],
        horizontal_spacing=hs,
        vertical_spacing=0.09,
        specs=specs
    )
    fig.update_annotations(font=dict(size=PT6))
    for anno in fig.layout.annotations:
        anno.update(y=anno.y+0.03)
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
            pos_std = values[2]

            # Negative
            neg_mean   = values[5]
            neg_median = values[6]
            neg_q25    = values[8]
            neg_q75    = values[9]
            neg_std = values[7]


            # --- Positive IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                #y=np.concatenate([pos_q25, pos_q75[::-1]]),
                y=np.concatenate([pos_mean+pos_std, (pos_mean-pos_std)[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Positive IQR",
                fillcolor=colors[0],
                line=dict(width=0),

            ), col=k*2+2, row=i+1)
            # --- Negative IQR band ---

            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                #y=np.concatenate([neg_q25, neg_q75[::-1]]),
                y=np.concatenate([neg_mean+neg_std, (neg_mean-neg_std)[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Negative IQR",
                fillcolor=colors[1],
                line=dict(width=0)
            ), col=k*2+2, row=i+1)

            # Positive median
            fig.add_trace(go.Scatter(
                x=x, y=pos_mean,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0], )
            ), col=k*2+2, row=i+1)
            fig.add_trace(go.Scatter(
                x=x, y=pos_mean,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0], )
            ), col=k*2+1, row=i+1)



            

            # Negative median
            fig.add_trace(go.Scatter(
                x=x, y=neg_mean,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1],)
            ), col=k*2+2, row=i+1)
            
            fig.add_trace(go.Scatter(
                x=x, y=neg_mean,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1], )
            ), col=k*2+1, row=i+1)

  


    fig.update_traces(showlegend=False)

    fig.update_xaxes(dtick=4000)
    fig.update_yaxes(
        col=1,
        title=dict(text="normalized signal", standoff=5)
    )
    fig.update_yaxes(
        col=3,
        title=dict(text="AlphaGenome score [AU]", standoff=5)
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
                line=dict(width=1, color="black"),
                name="Mean",
                marker=dict(size=0),
                mode="lines",



            ),

            go.Scatter(
                x=[None],
                y=[None],
                fill="toself",
                opacity=opacity,
                name="Std.",
                fillcolor="black",
                line=dict(width=0),
                marker=dict(size=0),
                mode="lines",

            )
        ]
    )
    fig.update_layout(
        template="simple_white_custom",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, yref="paper"),
        margin=dict(t=10, b=35, l=20, r=20),
        height=350,
        width=(WIDTH // 4) * 3
    )
    return fig


def plot_median_quantiles(prefix, outfile_postfix,  colors=["#1F77B4", "#FF7F0E"], opacity=0.4):
    """
    values: shape (10, N)
    Plot:
        - median (solid)
        - q25–q75 (filled band)
        - mean (dashed)
    """
    hs = 0.05
    extra = 0.02
    feats = [f for f in FEATURES if f.startswith("EX")]
    specs = [
        [
            {"l": extra},                      # col 1 - normal
            {"r": extra},            # col 2 - push right neighbour away
            {"l": extra},                      # col 3 - normal
            {"r": extra}                       # col 4 - normal
        ]
        for _ in range(len(feats))
    ]
    fig = make_subplots(
        rows=len(feats),
          cols=4, 
          x_title="Distance [nt]",
        #row_titles=[f[3:] for f in feats],
        horizontal_spacing=hs,
        vertical_spacing=0.09,
        specs=specs
    )
    fig.update_annotations(font=dict(size=PT6))
    for anno in fig.layout.annotations:
        anno.update(y=anno.y+0.03)
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

            # Positive median
            fig.add_trace(go.Scatter(
                x=x, y=pos_median,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0], )
            ), col=k*2+2, row=i+1)
            fig.add_trace(go.Scatter(
                x=x, y=pos_median,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0], )
            ), col=k*2+1, row=i+1)

            # Positive mean (dashed)
            fig.add_trace(go.Scatter(
                x=x, y=pos_mean,
                mode="lines",
                name="Positive Mean",
                line=dict(width=1, color=colors[0], dash="dot",)
            ), col=k*2+1, row=i+1)

            

            # Negative median
            fig.add_trace(go.Scatter(
                x=x, y=neg_median,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1],)
            ), col=k*2+2, row=i+1)
            
            fig.add_trace(go.Scatter(
                x=x, y=neg_median,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1], )
            ), col=k*2+1, row=i+1)

            # Negative mean (dashed)
            fig.add_trace(go.Scatter(
                x=x, y=neg_mean,
                mode="lines",
                name="Negative Mean",
                line=dict(width=1, color=colors[1], dash="dot",)
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
                name="Mean",
                marker=dict(size=0),
                mode="lines",



            ),
            go.Scatter(
                x=[None],
                y=[None],
                line=dict(width=1, color="black"),
                marker=dict(size=0),
                mode="lines",

                name="Median"

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

    fig.update_yaxes(range=[0.2, 1.9], row=1, col=2)
    fig.update_yaxes(range=[0.1, 1.4], row=2, col=2)
    fig.update_yaxes(range=[40, 100], row=2, col=4)

    fig.update_layout(
        template="simple_white_custom",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, yref="paper"),
        margin=dict(t=10, b=35, l=20, r=20),
        height=350,
        width=(WIDTH // 4) * 3
    )

    return fig


def plot_zoomed_insets(prefix, outfile_postfix, colors=["#1F77B4", "#FF7F0E"], opacity=0.4, zoom_range=11):
    """
    Create separate zoomed-in plots (x: -zoom_range to +zoom_range nt) for each subplot.
    Each subplot from the original grid becomes its own single plot SVG.
    """
    feats = [f for f in FEATURES if f.startswith("EX")]

    for i, m_feat in enumerate(feats):
        for k, feature in enumerate([m_feat, m_feat[3:]]):
            values = np.load(f"{prefix}{feature}_{outfile_postfix}")
            center_idx = values.shape[1] // 2

            # Slice data for zoomed view
            zoom_slice = slice(center_idx - zoom_range, center_idx + zoom_range + 1)
            zoom_x = np.arange(-zoom_range, zoom_range + 1)

            # Apply H3 scaling if needed
            if feature.startswith("H3"):
                zoom_x = zoom_x * 128

            # Positive
            pos_mean = values[0][zoom_slice]
            pos_median = values[1][zoom_slice]
            pos_q25 = values[3][zoom_slice]
            pos_q75 = values[4][zoom_slice]
            pos_std = values[2][zoom_slice]

            # Negative
            neg_mean = values[5][zoom_slice]
            neg_median = values[6][zoom_slice]
            neg_q25 = values[8][zoom_slice]
            neg_q75 = values[9][zoom_slice]
            neg_std = values[7][zoom_slice]

            # Determine y-axis title based on column (k=0: Experimental, k=1: AlphaGenome)
            y_title = "normalized signal" if k == 0 else "AlphaGenome score [AU]"

            # Create single plot figure
            fig = make_subplots(rows=1, cols=1)

            # --- Positive IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([zoom_x, zoom_x[::-1]]),
                y=np.concatenate([pos_q25, pos_q75[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Positive IQR",
                fillcolor=colors[0],
                line=dict(width=0),
            ), row=1, col=1)
            
            # --- Negative IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([zoom_x, zoom_x[::-1]]),
                y=np.concatenate([neg_q25, neg_q75[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Negative IQR",
                fillcolor=colors[1],
                line=dict(width=0)
            ), row=1, col=1)

            # Positive median
            fig.add_trace(go.Scatter(
                x=zoom_x, y=pos_median,
                mode="lines",
                name="Positive Median",
                line=dict(width=1, color=colors[0])
            ), row=1, col=1)

            # Positive mean




            # Negative median
            fig.add_trace(go.Scatter(
                x=zoom_x, y=neg_median,
                mode="lines",
                name="Negative Median",
                line=dict(width=1, color=colors[1],)
            ), row=1, col=1)



            fig.update_traces(showlegend=False)

            # Set x-axis range to zoomed view
            fig.update_xaxes(range=[-zoom_range, zoom_range], title=None, dtick=10)
            #fig.update_yaxes(title_text=y_title)

            fig.update_layout(
                template="simple_white_custom",
                margin=dict(t=0, b=15, l=15, r=3),
                height=50,
                width=50
            )

            # Save individual SVG
            source_name = "EX" if feature.startswith("EX") else "AlphaGenome"
            outfile_name = f"Figures/DistributionRes_zoomed_{m_feat}_{source_name}.svg"
            fig.write_image(outfile_name)
            print(f"Saved: {outfile_name}")



def plot_zoomed_insetsMean(prefix, outfile_postfix, colors=["#1F77B4", "#FF7F0E"], opacity=0.4, zoom_range=11):
    """
    Create separate zoomed-in plots (x: -zoom_range to +zoom_range nt) for each subplot.
    Each subplot from the original grid becomes its own single plot SVG.
    """
    feats = [f for f in FEATURES if f.startswith("EX")]

    for i, m_feat in enumerate(feats):
        for k, feature in enumerate([m_feat, m_feat[3:]]):
            values = np.load(f"{prefix}{feature}_{outfile_postfix}")
            center_idx = values.shape[1] // 2

            # Slice data for zoomed view
            zoom_slice = slice(center_idx - zoom_range, center_idx + zoom_range + 1)
            zoom_x = np.arange(-zoom_range, zoom_range + 1)

            # Apply H3 scaling if needed
            if feature.startswith("H3"):
                zoom_x = zoom_x * 128

            # Positive
            pos_mean = values[0][zoom_slice]
            pos_median = values[1][zoom_slice]
            pos_q25 = values[3][zoom_slice]
            pos_q75 = values[4][zoom_slice]
            pos_std = values[2][zoom_slice]

            # Negative
            neg_mean = values[5][zoom_slice]
            neg_median = values[6][zoom_slice]
            neg_q25 = values[8][zoom_slice]
            neg_q75 = values[9][zoom_slice]
            neg_std = values[7][zoom_slice]

            # Determine y-axis title based on column (k=0: Experimental, k=1: AlphaGenome)
            y_title = "normalized signal" if k == 0 else "AlphaGenome score [AU]"

            # Create single plot figure
            fig = make_subplots(rows=1, cols=1)

            # --- Positive IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([zoom_x, zoom_x[::-1]]),
                y=np.concatenate([pos_mean+pos_std, (pos_mean-pos_std)[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Positive Std",
                fillcolor=colors[0],
                line=dict(width=0),
            ), row=1, col=1)
            
            # --- Negative IQR band ---
            fig.add_trace(go.Scatter(
                x=np.concatenate([zoom_x, zoom_x[::-1]]),
                y=np.concatenate([neg_mean+neg_std, (neg_mean-neg_std)[::-1]]),
                fill="toself",
                opacity=opacity,
                name="Negative Std",
                fillcolor=colors[1],
                line=dict(width=0)
            ), row=1, col=1)

            # Positive median
            fig.add_trace(go.Scatter(
                x=zoom_x, y=pos_mean,
                mode="lines",
                name="Positive Mean",
                line=dict(width=1, color=colors[0])
            ), row=1, col=1)

            # Positive mean




            # Negative median
            fig.add_trace(go.Scatter(
                x=zoom_x, y=neg_mean,
                mode="lines",
                name="Negative Mean",
                line=dict(width=1, color=colors[1],)
            ), row=1, col=1)



            fig.update_traces(showlegend=False)

            # Set x-axis range to zoomed view
            fig.update_xaxes(range=[-zoom_range, zoom_range], title=None)
            #fig.update_yaxes(title_text=y_title)

            fig.update_layout(
                template="simple_white_custom",
                margin=dict(t=0, b=15, l=15, r=3),
                height=50,
                width=100
            )

            # Save individual SVG
            source_name = "EX" if feature.startswith("EX") else "AlphaGenome"
            outfile_name = f"Figures/DistributionRes_zoomed_Mean{m_feat}_{source_name}.svg"
            fig.write_image(outfile_name)
            print(f"Saved: {outfile_name}")


def plot_results(prefix, outfile):
    #fig = plot_mean_std(values)
    fig = plot_median_quantiles(outfile_postfix=outfile, prefix=prefix, colors=[COLORS["jaxpetrol"], COLORS["jaxgold"]])
    fig.write_image("Figures/DistributionRes.svg")
    fig.write_html("Figures/DistributionRes.html")
    
    fig = plot_mean_std(outfile_postfix=outfile, prefix=prefix, colors=[COLORS["jaxpetrol"], COLORS["jaxgold"]])
    fig.write_image("Figures/DistributionRes2.svg")
    fig.write_html("Figures/DistributionRes2.html")

    # Generate zoomed-in views for each feature
    plot_zoomed_insets(prefix=prefix, outfile_postfix=outfile, colors=[COLORS["jaxpetrol"], COLORS["jaxgold"]], zoom_range=256)
    plot_zoomed_insetsMean(prefix=prefix, outfile_postfix=outfile, colors=[COLORS["jaxpetrol"], COLORS["jaxgold"]], zoom_range=256)



if __name__ == "__main__":
    plot_results(outfile="SummaryNumpyArray.npy", prefix="GenomicSummary/")