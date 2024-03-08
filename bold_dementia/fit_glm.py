
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
from bold_dementia.data.volumes import add_volumes
from utils.saving import save_run

config = get_config()

def make_fc_data(maps_spec, model_spec):
    

    maps_path = Path(sys.argv[1]) # This is not great but whatever
    AD_matrices = joblib.load(maps_path / "AD.joblib")
    control_matrices = joblib.load(maps_path / "control.joblib")
    atlas = Atlas.from_name(maps_spec["ATLAS"], maps_spec["SOFT"])

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

    # If the model does not account for subjects, then they should be unique
    if model_spec["GROUPS"] != "sub": 
        df = df.groupby("sub").sample(n=1, random_state=config["seed"])
    else:
        print("Using several scans per subect and mixed models")

    # TODO Pass TIV only to be more efficient?
    if "total intracranial" in model_spec["RHS_FORMULA"]:
        print("Add intracranial volumes to phenotypes")
        df = add_volumes(df, config["volumes"])
    
    return df, edges, model_spec

# TODO Format report only, do not print, and move to utils
def display_df_report(df):
    print(f"Testing on {len(df)} subjects")
    print(f"{df.MA.sum()} AD")
    print(f"{df.DEMENCE_DAT.isna().sum()} controls")
    

# TODO Print info about the test, N, etc
def run_test(df, edges, model_spec):
    test_df = df.dropna(subset=["NIVETUD"])
    display_df_report(test_df)
    fit_df = lambda edge: fit_edges(
        edge, test_df, model_spec["RHS_FORMULA"], model_spec["GROUPS"]
    )
    parallel = Parallel(n_jobs=2, verbose=2)
    # TODO This is awfully slow
    # Change optimizer in statsmodel perhaps?
    with warnings.catch_warnings(category=ConvergenceWarning, action="ignore"):
        test_results = parallel(delayed(fit_df)(edge) for edge in edges)
    return test_results

    
import os
#os.environ['PYTHONWARNINGS']='ignore::ConvergenceWarning'

def main():
    maps_path = Path(sys.argv[1])
    model_specs_path = Path(sys.argv[2])
    
    maps_specs = get_config(maps_path / "parameters.yml")
    model_specs = get_config(model_specs_path)

    df, edges, parameters = make_fc_data(maps_specs, model_specs)
    print(df.head())

    results = run_test(df, edges, parameters) # TODO Chain functions
    stats, pvalues = zip(*results)
    _, pvalues_corr = fdrcorrection(pvalues)

    statmat = reshape_pvalues(stats)
    pmat = reshape_pvalues(pvalues_corr)
    matrix_export = {
        "statmap.joblib": statmat,
        "pmat.joblib": pmat
    }

    maps_name = maps_specs.pop("NAME")
    model_name = model_specs.pop("NAME")
    parameters = {**model_specs, **maps_specs}
    parameters["NAME"] = maps_name + "_" + model_name
    
    exppath = save_run(parameters, joblib.dump, matrix_export, dirkey="statresults")
    print(exppath)
    
if __name__ == "__main__":
    main()
    