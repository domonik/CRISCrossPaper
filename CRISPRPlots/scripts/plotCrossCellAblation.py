from plotIPSC import add_sig_bracket, significance_label
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from plotly_template import COLORS


def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    df = df[df["comparison"].isin(["CROSSCellMLM", "CrossCellType"])]
    breakpoint()
    mode_order = [
        "MLMFull-Ws512-AG_ccTesting",
        "MLMTCell-Ws512-AG_ccTesting",
        "MLMFull-Ws512+AG_ccTesting",
        "MLMTCell-Ws512+AG_ccTesting",
        "Arti-Ws23-AG_ccTesting",
        "Arti-Ws23+AG_ccTesting",
    ]
    df = df[df["mode"].isin(mode_order)]

    # Average AUCPR per mode, in the fixed order
    avg_df = (
        df.groupby("mode", as_index=False)["AUCPR"].mean()
        .set_index("mode").loc[mode_order].reset_index()
    )
    breakpoint()

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

    fig.write_html("Figures/CrossCellAblation.html")
    
    
    
if __name__ == "__main__":
    main()