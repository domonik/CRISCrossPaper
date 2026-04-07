import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import wilcoxon
from plotly_template import COLORS, WIDTH, PT8, PT6
from statsmodels.stats.multitest import multipletests
from plotly.subplots import make_subplots
import numpy as np

NAME_FIX = {
    "EX": "Ex",
    "CRISPRofft": "CRISPR-OFFT",
    "cnnCRISPR": "CnnCrispr",
}


def add_stars_with_stripes(tcell_results, strategy_order, tcell_df, fig, subplot_idx, strategy_col_prefix = "Strategy", space_fac = 0.15, show_ns=True):
    strategy_to_x = {s: i for i, s in enumerate(strategy_order)}
    y_max = tcell_df["mean"].max() + tcell_df["std"].max()
    line_spacing = space_fac * 0.15  # 5% of max for each level
    current_y = y_max + line_spacing
    for idx, row in tcell_results.iterrows():
        s1 = row[f"{strategy_col_prefix} 1"]
        s2 = row[f"{strategy_col_prefix} 2"]
        pval = row["padj"]

        # Decide stars
        if np.isnan(pval):
            continue
        elif pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"
        else:
            if show_ns:
                stars ="n.s."
            else:
                continue  # Not significant

        # x positions (center of bars)
        x0 = strategy_to_x[s1]
        x1 = strategy_to_x[s2]

        # Add horizontal line
        fig.add_shape(
            type="line",
            x0=x0, x1=x1,
            y0=current_y, y1=current_y,
            xref=f"x{subplot_idx}", yref=f"y{subplot_idx}",  # Column 1 corresponds to x1/y1
            line=dict(color="black", width=1)
        )

        # Add stars above the line
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=current_y if stars == "n.s." else current_y - 0.05,
            yanchor="bottom",
            text=stars,
            showarrow=False,
            xref=f"x{subplot_idx}",
            yref=f"y{subplot_idx}",
            font=dict(size=12) if "*" in stars else dict(size=PT6)
        )

        # Increment y for next line to avoid overlap
        current_y += line_spacing
    


def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def add_stars(fig, stats_df, pdf, modes, cat_col="window_size", x_col="mode", row=1, col=1):
    if cat_col in pdf:
        windows = list(pdf[cat_col].unique())
        use_windows = True

    else:
        windows = [1]
        use_windows = False
    n_groups = len(windows)
    group_width = 0.7
    box_width = group_width / n_groups
    for i, window in enumerate(windows):
        offset = -group_width/2 + box_width/2 + i * box_width

        for model in modes:
            if use_windows:
                prow = stats_df[
                    (stats_df[x_col] == model) &
                    (stats_df[cat_col] == int(window))
                ]
            else:
                prow = stats_df[
                    (stats_df[x_col] == model)
                ]
            if prow.empty or prow["padj"].values[0] >= 0.05:
                continue


            x_numeric = modes.index(model) + offset
            p = prow["padj"].values[0]
            fig.add_annotation(
                x=x_numeric,
                y=0.99,
                text=stars(p),
                showarrow=False,
                xref="x",
                yref="y domain",
                xanchor="center",
                font=dict(size=12),
                row=row,
                col=col
            )
            
            
def add_bar_values(fig, pdf, modes, row=1, col=1, cat_col="window_size", x_val="mode", outside = True, font_color="black", offset_x=0.02):
    """
    Add numeric values (e.g., 'mean') as annotations on top of each bar in a grouped bar chart.

    Parameters:
        fig : plotly Figure
            The figure to add annotations to.
        pdf : pd.DataFrame
            Dataframe containing 'mode', 'mean', 'window_size'.
        modes : list
            Ordered list of modes to match the x-axis.
        row, col : int
            Subplot row and column (default 1,1).
    """
    if cat_col in pdf:
        windows = list(pdf[cat_col].unique())
        use_windows = True

    else:
        windows = [1]
        use_windows = False
        
        
    n_groups = len(windows)
    group_width = 0.7
    box_width = group_width / n_groups

    for i, window in enumerate(windows):
        offset = -group_width/2 + box_width/2 + i * box_width - offset_x
        for model in modes:
            if use_windows:
                bar = pdf[(pdf[x_val] == model) &(pdf[cat_col] == window)]
            else:
                bar = pdf[
                    (pdf[x_val] == model)
                ]
            if bar.empty:
                continue

            mean_val = bar["mean"].values[0]
            x_numeric = modes.index(model) + offset
            fig.add_annotation(
                x=x_numeric,
                y=mean_val,           # place annotation at the top of the bar
                text=f"{mean_val:.3f}",  # formatted text
                showarrow=False,
                xref="x",
                yref="y",
                row=row,
                col=col,
                yanchor="bottom" if outside else "top",
                xanchor="center",
                textangle=90,
                font=dict(color=font_color)
            )


def calc_sig(pdf, base_model_stats):
    wilcoxon_results = []
    base_model = base_model_stats["name"].iloc[0]

    for (ws, mode, name), model_data in pdf.groupby(["window_size", "mode", "name"]):
        if name == base_model:
            continue


   
        # AG vs Base
        data = pd.concat((base_model_stats, model_data))
        data_pivot = data.pivot_table(index="split", columns='name', values='AUCPR')
        if ws == 512 and mode == "AG":
            breakpoint()
        if len(data_pivot) > 0:
            stat, p = wilcoxon(data_pivot[name], data_pivot[base_model])
                    

            wilcoxon_results.append({
                'mode': mode,
                "window_size": ws,
                'stat': stat,
                'p_value': p
            })


    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    _, wilcoxon_df["padj"], _, _ = multipletests(wilcoxon_df["p_value"])

    return wilcoxon_df
