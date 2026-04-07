import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly_template import COLORS, WIDTH

def main():
    df = pd.read_csv("validation_curves.tsv", sep="\t")
    df = df[df["mode"].str.contains("Arti-Ws512-Epi") == True]
    df = df[df["run"].str.contains("T_cell")]
    # df columns: ['run', 'step', 'metric', 'value']
    # Filter for validation AUPRC
    val_df = df[df['metric'] == 'val_auprc']

    # Group by step and compute statistics
    stats = val_df.groupby('step')['value'].agg(['mean', 'median', 'quantile'])
    # Better: compute IQR manually
    def iqr(x):
        return x.quantile(0.25), x.quantile(0.75)

    iqr_df = val_df.groupby('step')['value'].apply(iqr).apply(pd.Series)
    iqr_df.columns = ['q1', 'q3']

    # Merge stats
    step_stats = val_df.groupby('step')['value'].agg(['mean', 'median']).join(iqr_df)

    # Plot with Plotly
    fig = go.Figure()

    # Mean line
    fig.add_trace(go.Scatter(
        x=step_stats.index,
        y=step_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color=COLORS["jaxpetrol"])
    ))

    # IQR shaded area
    fig.add_trace(go.Scatter(
        x=list(step_stats.index) + list(step_stats.index[::-1]),
        y=list(step_stats['q3']) + list(step_stats['q1'][::-1]),
        fill='toself',
        fillcolor="rgba(0, 103, 120, 0.3)",
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='IQR'
    ))

    fig.update_layout(
        xaxis_title='Step',
        yaxis_title='Validation AUC-PR',
        xaxis=dict(range=[0, 300], dtick=50),
        template="simple_white_custom",
        height=200,
        width = WIDTH//2
    )
    max_steps = val_df.groupby('run')['step'].max() - 10
    mean_max_step = np.mean(max_steps)

    # Add vertical dashed line
    fig.add_vline(
        x=mean_max_step,
        line=dict(color='black', dash='dash'),
        annotation_text=f'Mean(best step){mean_max_step:.1f}',
        annotation_position='top right'
    )

    fig.write_image("Figures/ValidationCurve.svg")
    fig.write_html("Figures/ValidationCurve.html")
    

if __name__ == "__main__":
    main()