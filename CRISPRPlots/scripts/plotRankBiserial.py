import plotly.graph_objects as go
from plotly_template import COLORS, WIDTH
import pandas as pd

def main():
    
    results_df = pd.read_csv("Tables/FeatureMWUTable.tsv", sep="\t")
    results_df = results_df[results_df["significant"]]

    fig = go.Figure()
    for ft, subdf in results_df.groupby("feature_type"):

        color = COLORS["jaxpetrol"] if ft == "Experimental" else COLORS["jaxgold"]

        fig.add_trace(
            go.Violin(
                x=subdf["feature_type"],
                y=subdf["rank_biserial"].abs(),
                name=ft,
                line=dict(color="black", width=1),
                fillcolor=color,
                box_visible=True,
                meanline_visible=True,
                #points="all",          # show all individual points
                jitter=0.3,            # add horizontal spread
                marker=dict(size=3, color="black", opacity=1),
                showlegend=False,
                pointpos=0,
                spanmode="hard", 
                #orientation="h",      # horizontal violin!
            )
        )
    fig.update_layout(
        template="simple_white_custom",
        width=WIDTH // 4,
        height=350,

        margin=dict(l=35, r=20, t=50, b= 30),
        yaxis=dict(title="|Rank-biserial|")
    )
    fig.write_image("Figures/RankBiserialComparison.svg")
    fig.write_html("Figures/RankBiserialComparison.html")
    
if __name__ == "__main__":
    main()