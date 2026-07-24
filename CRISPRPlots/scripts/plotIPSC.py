import pandas as pd
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from plotly_template import COLORS


def significance_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def add_sig_bracket(fig, x1, x2, y, text):
    """Draw a bracket between x1 and x2 at height y with a label above it."""
    fig.add_shape(
        type="line",
        x0=x1, x1=x1, y0=y, y1=y + 0.02,
        line=dict(color="black", width=1.5),
    )
    fig.add_shape(
        type="line",
        x0=x2, x1=x2, y0=y, y1=y + 0.02,
        line=dict(color="black", width=1.5),
    )
    fig.add_shape(
        type="line",
        x0=x1, x1=x2, y0=y + 0.02, y1=y + 0.02,
        line=dict(color="black", width=1.5),
    )
    fig.add_annotation(
        x=(x1 + x2) / 2, y=y + 0.03,
        text=text, showarrow=False,
        font=dict(size=14),
    )

def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    df = df[df["comparison"] == "IPSC"]

    # Order so that compared modes (-AG vs +AG) are adjacent, grouped by Ws
    mode_order = [
        "Arti-Ws23-AG_ccTesting",
        "Arti-Ws23+AG_ccTesting",
        "Arti-Ws512-AG_ccTesting",
        "Arti-Ws512+AG_ccTesting",
    ]
    df = df[df["mode"].isin(mode_order)]

    # Average AUCPR per mode, in the fixed order
    avg_df = (
        df.groupby("mode", as_index=False)["AUCPR"].mean()
        .set_index("mode").loc[mode_order].reset_index()
    )

    colors = [
        COLORS["jaxgold"] if "+AG" in mode else COLORS["seagrey"]
        for mode in avg_df["mode"]
    ]

    fig = go.Figure(
        go.Bar(
            x=avg_df["mode"],
            y=avg_df["AUCPR"],
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Average AUCPR by Mode (IPSC)",
        yaxis_title="Average AUCPR",
        xaxis_title="Mode",
        yaxis_range=[0, 1.15],  # extra headroom for brackets
    )

    # --- Statistical tests: -AG vs +AG, done separately for Ws23 and Ws512 ---
    pairs = [
        ("Arti-Ws23-AG_ccTesting", "Arti-Ws23+AG_ccTesting"),
        ("Arti-Ws512-AG_ccTesting", "Arti-Ws512+AG_ccTesting"),
    ]

    for m1, m2 in pairs:
        vals1 = df.loc[df["mode"] == m1, "AUCPR"]
        vals2 = df.loc[df["mode"] == m2, "AUCPR"]

        stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        label = significance_label(p)

        x1 = mode_order.index(m1)
        x2 = mode_order.index(m2)
        y_max = max(
            avg_df.loc[avg_df["mode"] == m1, "AUCPR"].values[0],
            avg_df.loc[avg_df["mode"] == m2, "AUCPR"].values[0],
        )
        add_sig_bracket(fig, x1, x2, y_max + 0.05, f"{label} (p={p:.3g})")

    fig.write_html("Figures/IPSCs.html")

if __name__ == "__main__":
    main()