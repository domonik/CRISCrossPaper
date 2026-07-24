
import pandas as pd
import plotly.express as px

def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    df = df[df["comparison"] == "CRISPRATTANdK"]

    # Step 1: average AUCPR over seeds (averaging over splits within each seed first,
    # so each seed contributes one value per window_size/mode/dataset combo)
    grouped = (
        df.groupby(["window_size", "mode", "dataset", "seed"], as_index=False)["AUCPR"]
        .mean()
    )

    # grouped now has one row per (window_size, mode, dataset, seed) with mean AUCPR

    fig = px.box(
        grouped,
        x="window_size",
        y="AUCPR",
        color="mode",
        facet_col="dataset",
        points="all",
        title="Mean AUCPR (averaged over seeds) by window size, mode, and dataset",
    )

    fig.update_layout(boxmode="group")
    fig.show()

    # Optionally save
    fig.write_html("Figures/aucpr_boxplot.html")



if __name__ == "__main__":
    main()