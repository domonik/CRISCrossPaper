import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def main():
    file = "tensorboard_summaryFineTuningPaper3.tsv"
    df = pd.read_csv(file, sep="\t")
    print(df[df["test_auprc"].isna()])
    df.loc[df["mode"].isna(), "mode"] = "No Epi"
    breakpoint()
    df = df.groupby(['window_size', 'mode', 'split', "version", "dataset"], as_index=False)[['test_auprc', "best_val_loss", "nr_steps"]].mean()
    fig = px.box(
        df,
        x='mode',            # x-axis groups
        y='test_auprc',      # values to plot
        color='window_size',       # color by split
        points='all',        # show all data points
        title='Test AUPRC by Mode and Split',
        hover_data=['split', "best_val_loss", "nr_steps"],
        facet_row="dataset",
        labels={'test_auprc': 'Test AUPRC', 'mode': 'Mode', 'split': 'Split'}
    )
    fig.update_traces(boxmean=True)

    fig.update_layout(
        boxmode='group',     # group boxes per x-axis category
        xaxis_title='Mode',
        yaxis_title='Test AUPRC',
        legend_title='Window size'
    )
    fig.write_image("CrossAttnResultsPaper3.svg")
    fig.write_html("CrossAttnResultsPaper3.html")

    df_mean = (
    df
    .groupby(['mode', 'window_size', 'version'], as_index=False)
    .agg(
        mean_test_auprc=('test_auprc', 'mean'),
        mean_best_val_loss=('best_val_loss', 'mean'),
        nr_steps=("nr_steps", "mean")
    )
    
    )
    df_mean['window_size'] = df_mean['window_size'].astype(str)
    fig = px.scatter(
    df_mean,
    x='mean_best_val_loss',
    y='mean_test_auprc',
    color='mode',
    symbol='window_size',
    #facet_row="mode",
    labels={
        'mean_best_val_loss': 'Mean Best Val Loss',
        'mean_test_auprc': 'Mean Test AUPRC',
        'version': 'Version'
    },
    title='Mean Test AUPRC vs Mean Best Val Loss'
    )
    fig.write_html("CrossAttnScatter.html")


if __name__ == "__main__":
    main()
