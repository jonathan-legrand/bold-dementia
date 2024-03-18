import pandas as pd
import numpy as np

def pivot_resdf(resdf, alpha=0.05):
    resdf["node_a"] = resdf["edges"].map(lambda edge: edge.split("_")[0])
    resdf["node_b"] = resdf["edges"].map(lambda edge: edge.split("_")[1])

    mask = np.where(resdf["pvalues_fdr"] < alpha, 1, 0) 
    resdf["thresholded_beta"] = resdf["beta_AD"] * mask

    resdf = resdf.drop("edges", axis=1)

    resdf_pv = resdf.pivot(index="node_a" ,columns=["node_b"], values=["beta_AD", "pvalues_raw", "pvalues_fdr", "thresholded_beta"])
    return resdf_pv
    