import pandas as pd
import plotly.graph_objects as go
from scipy.stats import wilcoxon


def main():
    # --- Load data ---
    file = "AllExperiments_PerGuide.tsv"
    df = pd.read_csv(file, sep="\t")

    # --- Average over seeds for each (condition, GuideID) ---
    # Each row is one (seed, guide) pair. Grouping by condition+guide and
    # taking the mean gives one value per guide per condition
    # (12 values each for +AG and -AG).
    avg_df = df.groupby(["dataset", "condition", "GuideID"])["aucpr"].mean().reset_index()
    full_avg_df = df.groupby(["dataset", "condition"])["aucpr"].mean().reset_index()

    
    pooled_df = pd.read_csv("AllExperiments_PooledByCondition.tsv", sep="\t")
    full_avg_pooled_df = pooled_df.groupby(["dataset", "condition"])["aucpr"].mean().reset_index()

    breakpoint()

    # --- Paired Wilcoxon signed-rank test over guides ---
    plus_ag = avg_df[avg_df["condition"] == "+AG"].set_index("GuideID")["aucpr"]
    minus_ag = avg_df[avg_df["condition"] == "-AG"].set_index("GuideID")["aucpr"]

    # Align on GuideID so pairs match
    plus_ag, minus_ag = plus_ag.align(minus_ag, join="inner")

    stat, p_value = wilcoxon(plus_ag, minus_ag, zero_method="wilcox")
    print(f"Paired Wilcoxon test: V={stat:.1f}, p={p_value:.2e}")
    print(f"  +AG  : mean={plus_ag.mean():.4f}, std={plus_ag.std():.4f} (n={len(plus_ag)})")
    print(f"  -AG  : mean={minus_ag.mean():.4f}, std={minus_ag.std():.4f} KPer(n={len(minus_ag)})")

    # --- Significance stars ---
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # --- Aggregate for bar plot ---
    agg_df = avg_df.groupby("condition")["aucpr"].agg(["mean", "std"]).reset_index()

    conditions = agg_df["condition"].tolist()
    means = agg_df["mean"].tolist()
    stds = agg_df["std"].tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=conditions,
        y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=["#2E86AB", "#A23B72"],
    ))

    fig.update_layout(
        title="Cross-Cell Type K562 — AUCPR (averaged over seeds)",
        xaxis_title="Condition",
        yaxis_title="AUCPR",
        showlegend=False,
        height=400,
        width=500,
    )

    fig.add_annotation(
        x=0.5,
        y=max(means) + 0.02,
        text=f"p = {p_value:.2e} {stars}",
        showarrow=False,
        font=dict(size=12),
    )

    fig.write_html(
        "Figures/CrossCellTypeK562PerGuide.html",
    )
    print("Written to Figures/CrossCellTypeK562PerGuide.html")


if __name__ == "__main__":
    main()
