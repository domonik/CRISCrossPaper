import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import wilcoxon


def main():
    file = "FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    df = df[df["mode"].isin(["Arti-Ws512-AG_ccTesting", "Arti-Ws512+AG_ccTesting"])]
    df = df[~df["AUCPR"].isna()]
    agg_pdf = df.groupby(["mode"])["AUCPR"].agg(['mean', 'std']).reset_index()
    breakpoint()
    res = wilcoxon(df[df["mode"] == "Arti-Ws512+AG_ccTesting"]["AUCPR"], df[df["mode"] == "Arti-Ws512-AG_ccTesting"]["AUCPR"])
    print(res)
    

    fig = px.bar(
        agg_pdf,
        x='mode',
        y='mean',  # use mean for bar heights
        error_y='std',  # std will appear as error bars
        barmode='group',
    )
    fig.show()
    
    
    
if __name__ == "__main__":
    main()