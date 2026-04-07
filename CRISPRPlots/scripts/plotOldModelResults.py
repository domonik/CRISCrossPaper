import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from plotly_template import COLORS, WIDTH
from plotCrossAttnResults import add_bar_values
from statsmodels.stats.multitest import multipletests


NAME_FIX = {
    "crisprIP": "CRISPR-IP",
    "CRISPRofft": "CRISPR-OFFT",
    "cnnCRISPR": "CnnCrispr",
    "CRISPERT": "CRISPert",
    "CRISPROfft": "CRISPR-OFFT",
    "CRISPPert": "CRISPert"
}
custom_colors = {
        'AlphaGenome': COLORS['jaxgold'],
        'Sequence only': COLORS['white'],
        'Experimental': COLORS['jaxpetrol']
    }   

def main(df):
    df = df.groupby(['model', 'Feature', "pair"])['AUCPR'].mean().reset_index()
    df["model"] = df.model.map(lambda x: NAME_FIX[x] if x in NAME_FIX else x)
    wilcoxon_results = []
    for model_name, model_data in df.groupby('model'):
          
        # Ex vs Base
        ex_base = model_data[model_data['Feature'].isin(['Experimental','Sequence only'])]
        pivot_ex = ex_base.pivot_table(index="pair", columns='Feature', values='AUCPR')
        pivot_ex = pivot_ex.dropna(subset=['Experimental','Sequence only'])
        if len(pivot_ex) > 0:
            stat, p = wilcoxon(pivot_ex['Experimental'], pivot_ex['Sequence only'])
            wilcoxon_results.append({
                'model': model_name,
                'comparison': 'Ex vs Base',
                'stat': stat,
                'p_value': p
            })
        ex_base = model_data[model_data['Feature'].isin(['AlphaGenome','Sequence only'])]
        pivot_ex = ex_base.pivot_table(index="pair", columns='Feature', values='AUCPR')
        pivot_ex = pivot_ex.dropna(subset=['AlphaGenome','Sequence only'])
        if len(pivot_ex) > 0:
            stat, p = wilcoxon(pivot_ex['AlphaGenome'], pivot_ex['Sequence only'])
            wilcoxon_results.append({
                'model': model_name,
                'comparison': 'AG vs Base',
                'stat': stat,
                'p_value': p
            })

    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    _, wilcoxon_df["padj"], _, _  = multipletests(wilcoxon_df["p_value"])
    print(wilcoxon_df)
    # boxplot
    pdf = df.groupby(['model', 'Feature'])['AUCPR'].agg(["mean", "std"]).reset_index()
    desired_order =  ["Sequence only", "Experimental", "AlphaGenome"]
    pdf["Feature"] = pd.Categorical(pdf["Feature"], categories=desired_order, ordered=True)
    pdf = pdf.sort_values("Feature")
    fig = px.bar(
        pdf,
        x='model',
        y='mean',
        color='Feature', 
        color_discrete_map=custom_colors,
        category_orders={"Feature": desired_order}
        
        #boxmean=True
        # set custom colors        
        #barmode='group', 
        #error_y="std",              # group True/False next to each other
    )
    order = ["CRISPR-IP", "CnnCrispr", "CRISPR-OFFT", "CRISPert"]
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order,

    )
    
    add_bar_values(fig, pdf, modes=order, cat_col="Feature", x_val="model", offset_x=0)
    fig.update_traces(error_y=dict(thickness=1))

    fig.update_traces(

        marker_line_color='black',
        marker_line_width=1,
    )

    fig.update_layout(
        barmode='group',  # True/False side_features next to each other
        template="simple_white_custom",
        bargap=0.3,       # space between groups of bars (between models)
        bargroupgap=0.2,
        height=250,
        width = WIDTH // 2,
        legend=dict(orientation="h", y=1, title=None, yanchor="bottom", x=-0.1),
        margin=dict(l=50)
    )
    fig.update_xaxes(title=dict(text=None))
    fig.update_yaxes(title=dict(text="AUC-PR"))
    fig.write_html("Figures/OtherModels.html")
    fig.write_image("Figures/OtherModels.svg")


if __name__ == "__main__":
    file = "ex_vs_base.csv"
    df1 = pd.read_csv(file)
    df1 = df1.rename(
        {
            "NAME": "model",
            "AUPRC": "AUCPR",
            "group": "pair",
            
        }, axis=1
    )
    df1["Feature"] = df1["side_features"].apply(lambda x: "Experimental" if x else "Sequence only")
    df1 = df1.rename(
        {"auprc": "AUCPR"}, axis=1
    )
    
    file2 = "AG_vs_base.csv"
    df2 = pd.read_csv(file2)
    df2 = df2.rename(
        {
            "NAME": "model",
            "AUPRC": "AUCPR",
            "group": "pair",
            
        }, axis=1
    )
    df2["Feature"] = df2["side_features"].apply(lambda x: "AlphaGenome" if x else "Sequence only")
    df2 = df2[df2["Feature"] == "AlphaGenome"]
    df = pd.concat((df1, df2))
    df.to_csv("OldModlsfull.tsv", sep="\t", index=False)
    main(df)