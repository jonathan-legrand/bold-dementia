
import json
import sys
from pathlib import Path
import warnings
from bold_dementia import get_config

import pandas as pd
import joblib
from joblib import Parallel, delayed
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from bold_dementia.connectivity import Atlas, reshape_pvalues, vec_idx_to_mat_idx, z_transform_to_vec
from bold_dementia.models.matrix_GLM import fit_edges
from utils.saving import save_run

config = get_config()

def make_fc_data(maps_path):
    with open(maps_path / "parameters.json", "r") as stream:
        parameters = json.load(stream)
    print(parameters)

    AD_matrices = joblib.load(maps_path / "AD.joblib")
    control_matrices = joblib.load(maps_path / "control.joblib")
    atlas = Atlas.from_name(parameters["ATLAS"], parameters["SOFT"])

    AD_df = pd.read_csv(maps_path / "balanced_AD.csv", index_col=0)
    control_df = pd.read_csv(maps_path / "balanced_control.csv", index_col=0)
    df = pd.concat((AD_df, control_df))

    
    AD_vec = np.array([z_transform_to_vec(mat) for mat in AD_matrices])
    control_vec = np.array([z_transform_to_vec(mat) for mat in control_matrices])

    fc = np.vstack((AD_vec, control_vec))
    l = fc.shape[1]
    labels = atlas.labels
    rows, cols = vec_idx_to_mat_idx(l)
    edges = [f"{labels[i]}_{labels[j]}" for i, j in zip(rows, cols)]

    fc = pd.DataFrame(fc, columns=edges)

    df["AD"] = np.where(df.scan_to_onset < 0, 1, 0)
    df = pd.concat([df.reset_index(drop=True), fc], axis=1, join="inner")
    df = df.drop(df[df.MA == 0].index) # Drop MA == 0

    if parameters["LONGITUDINAL"] is False:
        df = df.groupby("sub").sample(n=1, random_state=config["seed"])
        print(df.head())
    
    return df, edges, parameters

# TODO Choose formula in run_config?
def run_test(df, edges):
    test_df = df.dropna(subset=["NIVETUD"])
    fit_df = lambda edge: fit_edges(edge, test_df)
    parallel = Parallel(n_jobs=2, verbose=2)
    # TODO This is awfully slow
    # Change optimizer in statsmodel perhaps?
    with warnings.catch_warnings(category=ConvergenceWarning, action="ignore"):
        test_results = parallel(delayed(fit_df)(edge) for edge in edges)
    return test_results

    
import os
os.environ['PYTHONWARNINGS']='ignore::ConvergenceWarning'

def main():
    maps_path = Path(sys.argv[1])
    df, edges, parameters = make_fc_data(maps_path)

    results = run_test(df, edges)
    stats, pvalues = zip(*results)
    _, pvalues_corr = fdrcorrection(pvalues)

    statmat = reshape_pvalues(stats)
    pmat = reshape_pvalues(pvalues_corr)
    matrix_export = {
        "statmap.joblib": statmat,
        "pmat.joblib": pmat
    }
    save_run(parameters, joblib.dump, matrix_export)
    

if __name__ == "__main__":
    main()
    