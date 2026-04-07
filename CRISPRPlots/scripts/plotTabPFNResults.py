import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import plotly.express as px
from plotly_template import COLORS, WIDTH, SINGLE_COL
import plotly.graph_objects as go
from helpers import add_bar_values, add_stars

def calc_sig(pdf, base_model_stats):
    wilcoxon_results = []
    base_model = base_model_stats["Epi"].iloc[0]

    for (name, ), model_data in pdf.groupby(["Epi"]):
        if name == base_model:
            continue


   
        # AG vs Base
        data = pd.concat((base_model_stats, model_data))
        data_pivot = data.pivot_table(index="split", columns='Epi', values='AUCPR')
        if len(data_pivot) > 0:
            stat, p = wilcoxon(data_pivot[name], data_pivot[base_model])

                    

            wilcoxon_results.append({
                'Epi': name,
                'stat': stat,
                'p_value': p
            })


    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    _, wilcoxon_df["padj"], _, _ = multipletests(wilcoxon_df["p_value"])

    return wilcoxon_df


def main(tsv):
    df = pd.read_csv(tsv, sep="\t")
    df = df.groupby(["Epi", "split"])["AUCPR"].mean().reset_index()
    base_model = df[df["Epi"] == "Baseline"]
    wildf = calc_sig(df, base_model)
    mdf = df.groupby(["Epi"])["AUCPR"].agg(['mean', 'std']).reset_index()
    fig = go.Figure()
    cmap = {
        "Baseline": COLORS["white"],
        "AG": COLORS["jaxgold"],
        "EX": COLORS["jaxpetrol"],
    }
    for _, row in mdf.iterrows():
        fig.add_bar(
            x=[row["Epi"]],
            y=[row["mean"]],
            #error_y=dict(array=[row["std"]]),
            marker_color=cmap[row["Epi"]],
            marker_line=dict(color="black", width=1),
            showlegend=False,
            width=0.4
        )
    fig.update_layout(
        template="simple_white_custom",
        width = WIDTH // 3,
        height= 200,
        margin=dict(b=25, l=50)
    )
    modes = ["Baseline", "EX", "AG"]
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=modes,
        ticktext=["Baseline", "Experimental", "AlphaGenome"],
    )
    mdf["mode"] = mdf["Epi"]
    add_bar_values(fig, mdf, row=None, col=None, modes=modes)
    add_stars(fig, wildf, mdf, modes=modes, x_col="Epi", col=None, row=None)
    fig.update_yaxes(range=[0.7, 0.8], title="AUC-PR")
    fig.write_html("Figures/TabPFNResults.html")
    fig.write_image("Figures/TabPFNResults.svg", width=fig.layout.width, height=fig.layout.height)



if __name__ == "__main__":
    
    file ="TabPFNResults.tsv"
    main(file)