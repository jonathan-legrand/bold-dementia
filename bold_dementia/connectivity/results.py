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
    
def edges_to_matrix(df:pd.DataFrame, **pivot_kwargs):
    """
    Transform edge values into a full, symmetric, node to node comparison matrix
    """
    df["node_a"] = df["edges"].map(lambda edge: edge.split("_")[0])
    df["node_b"] = df["edges"].map(lambda edge: edge.split("_")[1])
    swapped = df.rename(columns={"node_a": "node_b", "node_b": "node_a"})
    symmetric_closure = pd.concat((df, swapped))
    pv = symmetric_closure.pivot(index="node_a", columns=["node_b"], **pivot_kwargs)
    return pv