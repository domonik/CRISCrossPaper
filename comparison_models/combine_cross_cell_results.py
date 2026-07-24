
import pandas as pd

def load_model_results(csv_path, name, cell_type="k562"):
    df = pd.read_csv(csv_path)
    df["name"] = name
    df["cell_type"] = cell_type
    return df


# hard-coded input paths
crispert_path   = "CRISPert_cross_cell_results_fixed_k562.csv"
cnncrispr_path  = "cnnCRISPR_cross_cell_results_fixed_k562.csv"
crisprip_path   = "crisprIP_cross_cell_results_fixed_k562.csv"
crisprofft_path = "CRISPROfft_cross_cell_results_fixed_k562.csv"

# load data
crispert   = load_model_results(crispert_path, "CRISPert")
cnncrispr  = load_model_results(cnncrispr_path, "cnnCRISPR")
crisprip   = load_model_results(crisprip_path, "crisprIP")
crisprofft = load_model_results(crisprofft_path, "CRISPROfft")

# combine
combined_df = pd.concat(
    [crispert, cnncrispr, crisprip, crisprofft],
    ignore_index=True
)

combined_df.to_csv("../Results/combined_cross_cell_results.csv", index=False)




