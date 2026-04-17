import pickle
from plotly_shap_plotter import PlotlySHAPVisualizer
import numpy as np
import glob
import os 
from plotly_template import COLORS, WIDTH
import plotly.graph_objects as go



feature_name_map = {
    # Mismatch patterns at each guide position
    "MM_pattern_0": "Mismatch at Pos 0",
    "MM_pattern_1": "Mismatch at Pos 1",
    "MM_pattern_2": "Mismatch at Pos 2",
    "MM_pattern_3": "Mismatch at Pos 3",
    "MM_pattern_4": "Mismatch at Pos 4",
    "MM_pattern_5": "Mismatch at Pos 5",
    "MM_pattern_6": "Mismatch at Pos 6",
    "MM_pattern_7": "Mismatch at Pos 7",
    "MM_pattern_8": "Mismatch at Pos 8",
    "MM_pattern_9": "Mismatch at Pos 9",
    "MM_pattern_10": "Mismatch at Pos 10",
    "MM_pattern_11": "Mismatch at Pos 11",
    "MM_pattern_12": "Mismatch at Pos 12",
    "MM_pattern_13": "Mismatch at Pos 13",
    "MM_pattern_14": "Mismatch at Pos 14",
    "MM_pattern_15": "Mismatch at Pos 15",
    "MM_pattern_16": "Mismatch at Pos 16",
    "MM_pattern_17": "Mismatch at Pos 17",
    "MM_pattern_18": "Mismatch at Pos 18",
    "MM_pattern_19": "Mismatch at Pos 19",
    "MM_pattern_20": "Mismatch at Pos 20",
    "MM_pattern_21": "Mismatch at Pos 21",
    "MM_pattern_22": "Mismatch at Pos 22",

    # Mismatch count
    "distance": "Number of Mismatches",

    # Free energy terms
    "dG_CRISPRoff_Total": "CRISPRoff Score",
    "dG_RNA_DNA_weighted_total": "RNA-DNA ΔG (weighted)",
    "dG_RNA_RNA_fold": "gRNA Folding ΔG",
    "dG_DNA_DNA_open": "DNA Opening ΔG",

    # Energy terms
    "E_RNA_DNA": "RNA-DNA Energy",
    "E_corr_RNA_DNA": "Corrected RNA-DNA Energy",
    "E_gRNA_fold": "gRNA Folding Energy",

    # Position-specific energies
    "E_corr_RNA_DNA_pos1": "Energy Pos 1",
    "E_corr_RNA_DNA_pos2": "Energy Pos 2",
    "E_corr_RNA_DNA_pos3": "Energy Pos 3",
    "E_corr_RNA_DNA_pos4": "Energy Pos 4",
    "E_corr_RNA_DNA_pos5": "Energy Pos 5",
    "E_corr_RNA_DNA_pos6": "Energy Pos 6",
    "E_corr_RNA_DNA_pos7": "Energy Pos 7",
    "E_corr_RNA_DNA_pos8": "Energy Pos 8",
    "E_corr_RNA_DNA_pos9": "Energy Pos 9",
    "E_corr_RNA_DNA_pos10": "Energy Pos 10",
    "E_corr_RNA_DNA_pos11": "Energy Pos 11",
    "E_corr_RNA_DNA_pos12": "Energy Pos 12",
    "E_corr_RNA_DNA_pos13": "Energy Pos 13",
    "E_corr_RNA_DNA_pos14": "Energy Pos 14",
    "E_corr_RNA_DNA_pos15": "Energy Pos 15",
    "E_corr_RNA_DNA_pos16": "Energy Pos 16",
    "E_corr_RNA_DNA_pos17": "Energy Pos 17",
    "E_corr_RNA_DNA_pos18": "Energy Pos 18",
    "E_corr_RNA_DNA_pos19": "Energy Pos 19",
    "E_corr_RNA_DNA_pos20": "Energy Pos 20",

    # PAM
    "PAM_Ratio": "PAM Ratio",

    # Chromatin accessibility
    "atac_23bp_median_AG": "ATAC 23bp Median",
    "atac_4096bp_median_AG": "ATAC 4096bp Median",

    # Histone marks
    "H3K4me1_256bp_min_AG": "H3K4me1 256bp Min",
    "H3K4me3_8192bp_mean_Ag": "H3K4me3 8192bp Mean",
    "H3K36me3_8192bp_mean_Ag": "H3K36me3 8192bp Mean",
}



def main():
    base_dir = "shap"
    all_values_list = []
    all_data_list = []

    # Find all sgX directories
    sg_dirs = sorted(glob.glob(os.path.join(base_dir, "sg*")))

    for sg_dir in sg_dirs:
        pos_path = os.path.join(sg_dir, "shap_positive.pkl")
        neg_path = os.path.join(sg_dir, "shap_negative.pkl")

        if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
            continue

        with open(pos_path, "rb") as f:
            shap_pos = pickle.load(f)

        with open(neg_path, "rb") as f:
            shap_neg = pickle.load(f)

        # Extract class 1 and stack within group
        values = np.vstack([
            shap_pos.values[:, :, 1],
            shap_neg.values[:, :, 1]
        ])
        data = np.vstack([
            shap_pos.data,
            shap_neg.data
        ])

        all_values_list.append(values)
        all_data_list.append(data)

    # Stack across all sgX
    all_values = np.vstack(all_values_list)
    all_data = np.vstack(all_data_list)

    feature_name_array = np.array(shap_pos.feature_names)                                                                                                                      
    renamed_features = np.array([feature_name_map.get(f, f) for f in feature_name_array]) 
    # Create combined Explanation
    combined = type('Explanation', (), {
        'values': all_values,
        'data': all_data,
        'base_values': shap_pos.base_values,
        'feature_names': renamed_features
    })()
    
    viz = PlotlySHAPVisualizer(combined)

    # Define discrete features - these will use categorical coloring instead of outlier-based coloring
    # distance takes discrete values 1-6, and MM_pattern_* features are also discrete
    discrete_features = [
        "distance",  # Discrete values 1-6 (mismatch count)
        "MM_pattern_15",
        "MM_pattern_21",
        "MM_pattern_16",
        "MM_pattern_13",
        "MM_pattern_20",
        "MM_pattern_6",
        "MM_pattern_19",
        "MM_pattern_4",
        "MM_pattern_18",
        "MM_pattern_12",
        "MM_pattern_11",
        "MM_pattern_7",
        "MM_pattern_3",
        "MM_pattern_10",
        "MM_pattern_5",
        "MM_pattern_2",
        "MM_pattern_8",
        "MM_pattern_9",
    ]

    # Define colorscale for continuous/binary features: jaxpetrol to jaxgold
    colorscale_norm = [[0, COLORS["jaxpetrol"]], [1, COLORS["jaxgold"]]]
    colorscale = [COLORS["vikpurple"], COLORS["bufblue"], COLORS["miablue"], COLORS["seagreen"], COLORS["miaorange"], COLORS["bufred"], ]
    use_features_renamed = [feature_name_map.get(f, f) for f in [
      "distance",
      "dG_CRISPRoff_Total",
      "MM_pattern_15",
      "atac_23bp_median_AG",
      "H3K4me3_8192bp_mean_Ag",
      "H3K4me1_256bp_min_AG",
      "atac_4096bp_median_AG",
      "H3K36me3_8192bp_mean_Ag"
    ]]
    discrete_features_renamed = [feature_name_map.get(f, f) for f in [
      "distance",
      "MM_pattern_15", "MM_pattern_21", "MM_pattern_16", "MM_pattern_13",
      "MM_pattern_20", "MM_pattern_6", "MM_pattern_19", "MM_pattern_4",
      "MM_pattern_18", "MM_pattern_12", "MM_pattern_11", "MM_pattern_7",
      "MM_pattern_3", "MM_pattern_10", "MM_pattern_5", "MM_pattern_2",
      "MM_pattern_8", "MM_pattern_9",                                                                                                                                        
    ]] 
    
    fig1 = viz.summary_plot(colorscale_continuous=colorscale_norm, colorscale=colorscale,
                            max_features=0, max_jitter=0.45, use_features=use_features_renamed,
                            discrete_features=discrete_features_renamed,
                            )
    fig1.update_layout(
        template="simple_white_custom",
        width=WIDTH//3 * 2,
        height=200,
        margin=dict(r=20, l=100, b=40, t=20)
    )
    fig1.update_traces(
        marker=dict(size=3),
    )
    for i in range(1, 7):
        fig1.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                marker=dict(color=colorscale[i-1]),
                name=int(i),
                showlegend=True
                
            )
        )
    fig1.update_layout(
        coloraxis=dict(
            showscale=False
        ),
        coloraxis2=dict(
            colorbar=dict(len=1, x=1.22,
            xanchor="left",),
           
        ),
        legend=dict(title=dict(text="Raw value"), y=0.945, yanchor="top")
    )
  
    fig1.write_image("Figures/shapVals.svg",)
    fig1.write_image("Figures/shapVals.png", width=WIDTH//3 * 2, height=200, scale=3)
    fig1.write_html("Figures/shapVals.html")
    fig2 = viz.summary_plot(colorscale_continuous=colorscale_norm,
                           max_features=0, max_jitter=0.45, discrete_features=discrete_features,
                           )
    fig2.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=700,
        margin=dict(r=20, l=200)
    )
    fig2.update_traces(marker=dict(size=3))
    fig2.write_image("Figures/shapValsSupplement.svg")
    fig2.write_image("Figures/shapValsSupplement.png", width=WIDTH, height=700, scale=2)
    fig2.write_html("Figures/shapValsSupplement.html")
    
    df = viz.get_df()
    df = df.sort_values("MeanAbsSHAP", ascending=False).reset_index()
    df.to_csv("Tables/SHAPSummary.tsv", sep="\t")


if __name__ == "__main__":
    main()