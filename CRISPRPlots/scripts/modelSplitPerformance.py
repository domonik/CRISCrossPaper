import pandas as pd
import plotly.express as px
from plotOldModelResults import custom_colors, NAME_FIX


splits = [['sg28', 'sg23', 'sg3'], ['sg2', 'sg24', 'sg8'], ['sg1', 'sg6', 'sg7'], ['sg12', 'sg18', 'sg10'], ['sg13', 'sg26', 'sg5'], ['sg19', 'sg16']]
SPLITS = [g for glist in splits for g in glist]



def main(df):
    df = df.groupby(['model', 'window_size', "split", "Features"], dropna=False)['AUCPR'].std().reset_index(name='AUCPR_std')
    df = df.groupby(['model', 'window_size', "Features"], dropna=False)["AUCPR_std"].mean().reset_index()
    df["model window"] = df["model"] + " (" + df["window_size"].astype(str) + ")"
    fig = px.bar(
        df,
        x="model window",
        y="AUCPR_std",
        color="Features",
        color_discrete_map=custom_colors,
            barmode="group"
        #points="all",
        #title="AUROC Standard Deviation per Model per Split"
    )
    # for trace in fig.data:
    #     trace.fillcolor = trace.marker.color
    #     trace.opacity = 1
    #     trace.line.color = 'black'
    #     trace.line.width = 1
    # fig.update_traces(
    #     line=dict(color='black', width=1),
    #     opacity=1,
    #     boxmean=True,
    #         pointpos=0,
    #         marker=dict(size=1),
    #     marker_line_color='black',
    #     marker_color="black",
    #     marker_size=2

    # )
    fig.update_traces(
        marker=dict(line=dict(color="black", width=1))
    )
    fig.update_layout(
        bargap=0.3,       # space between groups of bars (between models)
        bargroupgap=0.2,
        template="simple_white_custom",
        height=300,
        xaxis=dict(title=""),
        yaxis=dict(title="\u03C3(AUC-PR)"),
    )
    fig.write_html("Figures/PredictionStdev.html")
    fig.write_image("Figures/PredictionStdev.svg")


if __name__ == "__main__":
    other_models = "OldModlsfull.tsv"
    cross_res = "tensorboard_summaryFineTuningPaper3.tsv"
    tabpfn_res = "TabPFNResults.tsv"
    
    df3 = pd.read_csv(tabpfn_res, sep="\t")


    df1 = pd.read_csv(other_models, sep="\t")
    df1["model"] = df1.model.map(lambda x: NAME_FIX[x] if x in NAME_FIX else x)

    df2 = pd.read_csv(cross_res, sep="\t")
    df2.loc[df2["mode"].isna(), "mode"] = "Base"
    df2 = df2[df2["dataset"] == "T-Cell"]

    df2["split"] = df2["split"].map(lambda y: SPLITS[y])
    df2["model"] = "CrossAttention"
    df2["mode"] = df2["mode"].map(lambda x: "Ex" if x == "EX" else x)

    df2 = df2[df2["mode"].isin(["Base", "AG", "Ex"])]
    modemap = {
        "AG": "AlphaGenome",
        "Ex": "Experimental",
        "Base": "Sequence only"
    }

    df2["mode"] = df2["mode"].map(modemap)
    df2 = df2.rename({"test_auprc": "AUCPR", "mode": "Features"}, axis=1)
    df1["window_size"] = 2 ** 13
    df1 = df1.rename({"Feature": "Features", "pair": "split", "auprc": "AUCPR"}, axis=1)

    df2 = df2[["model", "window_size", "Features", "split", "seed", "AUCPR"]]
    df1 = df1[["model", "window_size", "Features", "split", "seed", "AUCPR"]]
    df3 = df3.rename(
        {"Epi": "Features",
         },
        axis=1
    )
    df3["window_size"] = 2 ** 13
    df3["model"] = "TabPFN"
    df3["Features"] = df3["Features"].map(lambda x: "Ex" if x == "EX" else "Base" if x == "Baseline" else x)

    df3 = df3[["model", "window_size", "Features", "split", "seed", "AUCPR"]]
    df3["Features"] = df3["Features"].map(modemap)

    df = pd.concat((df1, df2, df3))

    main(df)