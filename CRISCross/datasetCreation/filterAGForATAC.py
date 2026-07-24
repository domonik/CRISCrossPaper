import pandas as pd

def main():
    ag = pd.read_csv("../datasets/AlphagenomeInfo.csv", sep="\t")
    atac_ag = ag[(ag["output_type"] == "ATAC") & (ag["organism"] == "human")]
    print(atac_ag[atac_ag["biosample_name"] == "WTC11"])
    
    
    


if __name__ == "__main__":
    main()