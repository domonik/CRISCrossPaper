import pickle
from plotly_shap_plotter import PlotlySHAPVisualizer
import numpy as np
import glob
import os 
from plotly_template import COLORS, WIDTH



feature_name_map = {
    "distance": "Nr. mismatches",
    "dG_CRISPRoff_Total": "CRISPRoff Score (-\u0394G)",
    "E_corr_RNA_DNA": "E_corr RNA DNA",
    "MM_pattern_15": "MM Pattern 15",
    "MM_pattern_21": "MM Pattern 21",
    "E_corr_RNA_DNA_pos5": "E_corr RNA DNA pos5",
    "dG_RNA_DNA_weighted_total": "dG RNA DNA Weighted Total",
    "E_RNA_DNA": "E RNA DNA",
    "MM_pattern_16": "MM Pattern 16",
    "E_corr_RNA_DNA_pos14": "E_corr RNA DNA pos14",
    "MM_pattern_13": "MM Pattern 13",
    "MM_pattern_20": "MM Pattern 20",
    "MM_pattern_6": "MM Pattern 6",
    "E_corr_RNA_DNA_pos2": "E_corr RNA DNA pos2",
    "MM_pattern_19": "MM Pattern 19",
    "dG_DNA_DNA_open": "dG DNA DNA open",
    "PAM_Ratio": "PAM Ratio",
    "MM_pattern_4": "MM Pattern 4",
    "MM_pattern_18": "MM Pattern 18",
    "MM_pattern_12": "MM Pattern 12",
    "E_corr_RNA_DNA_pos10": "E_corr RNA DNA pos10",
    "E_corr_RNA_DNA_pos4": "E_corr RNA DNA pos4",
    "MM_pattern_0": "MM Pattern 0",
    "E_corr_RNA_DNA_pos3": "E_corr RNA DNA pos3",
    "E_corr_RNA_DNA_pos20": "E_corr RNA DNA pos20",
    "MM_pattern_11": "MM Pattern 11",
    "MM_pattern_7": "MM Pattern 7",
    "atac_23bp_median_AG": "ATAC 23bp Median AG",
    "atac_4096bp_median_AG": "ATAC 4096bp Median AG",
    "MM_pattern_3": "MM Pattern 3",
    "E_corr_RNA_DNA_pos11": "E_corr RNA DNA pos11",
    "MM_pattern_10": "MM Pattern 10",
    "MM_pattern_5": "MM Pattern 5",
    "dG_RNA_RNA_fold": "dG RNA RNA fold",
    "E_gRNA_fold": "E gRNA fold",
    "MM_pattern_2": "MM Pattern 2",
    "E_corr_RNA_DNA_pos17": "E_corr RNA DNA pos17",
    "MM_pattern_8": "MM Pattern 8",
    "MM_pattern_9": "MM Pattern 9",
    "E_corr_RNA_DNA_pos16": "E_corr RNA DNA pos16",
    "E_corr_RNA_DNA_pos12": "E_corr RNA DNA pos12",
    "E_corr_RNA_DNA_pos15": "E_corr RNA DNA pos15",
    "E_corr_RNA_DNA_pos18": "E_corr RNA DNA pos18",
    "E_corr_RNA_DNA_pos19": "E_corr RNA DNA pos19",
    "H3K4me3_8192bp_mean_Ag": "H3K4me3 8192bp Mean AG",
    "H3K4me3_256bp_min_AG": "H3K4me3 256bp Min AG",
    "H3K4me1_256bp_min_AG": "H3K4me1 256bp Min AG",
    "E_corr_RNA_DNA_pos1": "E_corr RNA DNA pos1",
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

    # Create combined Explanation
    combined = type('Explanation', (), {
        'values': all_values,
        'data': all_data,
        'base_values': shap_pos.base_values,
        'feature_names': shap_pos.feature_names
    })()
    
    viz = PlotlySHAPVisualizer(combined)
    fig1 = viz.summary_plot(colorscale=[COLORS["jaxgold"], COLORS["jaxpetrol"]], max_features=0, max_jitter=0.45,
                            use_features=[
                                "distance", 
                                "dG_CRISPRoff_Total",
                                "MM_pattern_15",
                                "atac_23bp_median_AG",
                                "H3K4me3_8192bp_mean_Ag",
                                "H3K4me1_256bp_min_AG",
                                "atac_4096bp_median_AG",
                                "H3K36me3_8192bp_mean_Ag"
                                ])
    fig1.update_layout(
        template="simple_white_custom",
        width=WIDTH//3 * 2,
        height=200,
        margin=dict(r=20, l=100, b=25),
        coloraxis_colorbar_title="Norm. value"
    )
    fig1.update_traces(
        marker=dict(size=3),
        
        )
    current_ticktext = fig1.layout.yaxis.ticktext
    patched_ticktext = [feature_name_map.get(lbl, lbl) for lbl in current_ticktext]

    fig1.update_yaxes(ticktext=patched_ticktext)
    
    
    fig1.write_image("Figures/shapVals.svg")
    fig1.write_html("Figures/shapVals.html")
    
    fig2 = viz.summary_plot(colorscale=[COLORS["jaxgold"], COLORS["jaxpetrol"]], max_features=0, max_jitter=0.45, )
    fig2.update_layout(
        template="simple_white_custom",
        width=WIDTH,
        height=700,
        margin=dict(r=20, l=100),
        coloraxis_colorbar_title="Norm. value"
    )
    current_ticktext = fig2.layout.yaxis.ticktext
    patched_ticktext = [feature_name_map.get(lbl, lbl) for lbl in current_ticktext]
    fig2.update_layout(coloraxis=dict(cmin=0, cmax=1))
    #fig2.update_yaxes(ticktext=patched_ticktext)
    fig2.update_traces(marker=dict(size=2))
    fig2.write_image("Figures/shapValsSupplement.svg")
    fig2.write_html("Figures/shapValsSupplement.html")
    
    df = viz.get_df()
    df = df.sort_values("MeanAbsSHAP", ascending=False).reset_index()
    df.to_csv("Tables/SHAPSummary.tsv", sep="\t")


if __name__ == "__main__":
    main()