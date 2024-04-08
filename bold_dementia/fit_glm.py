
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

from bold_dementia.connectivity import (
    Atlas, reshape_pvalues, vec_idx_to_mat_idx, z_transform_to_vec, group_groupby, edge_format
)
from bold_dementia.models.matrix_GLM import fit_edges
from bold_dementia.data.volumes import add_volumes
from utils.saving import save_run
from utils.iterables import join, all_connectivities
from bold_dementia.stats.prepare_individuals import make_fc_data

config = get_config()



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
    # TODO check n_jobs in config and set a default value
    parallel = Parallel(n_jobs=8, verbose=2)

    # TODO This is awfully slow for mixed models and has convergence issues
    # Change optimizer in statsmodel perhaps?
    with warnings.catch_warnings(category=ConvergenceWarning, action="ignore"):
        test_results = parallel(delayed(fit_df)(edge) for edge in edges)
    return test_results

    
import os
#os.environ['PYTHONWARNINGS']='ignore::ConvergenceWarning'

def main():
    maps_path = Path(config["connectivity_matrices"]) / sys.argv[1]
    model_specs_path = Path(sys.argv[2])
    
    maps_specs = get_config(maps_path / "parameters.yml")
    model_specs = get_config(model_specs_path)
    print(model_specs)

    df, edges, parameters = make_fc_data(maps_path, maps_specs, model_specs)
    print(df.head())

    results = run_test(df, edges, parameters) # TODO Chain functions
    stats, pvalues = zip(*results)

    print(f"Correcting FDR with {len(pvalues)} comparisons")
    _, pvalues_corr = fdrcorrection(pvalues)

    maps_name = maps_specs.pop("NAME")
    model_name = model_specs.pop("NAME")
    parameters = {**model_specs, **maps_specs}
    parameters["NAME"] = maps_name + "_" + model_name
    
    # The original matrix ordering is preserved 
    # when no groupby happened,
    # otherwise it's better to just provide resdf for further
    # analysis
    if "BY_BLOCK" not in model_specs.keys() or model_specs["BY_BLOCK"] is False:
        statmat = reshape_pvalues(stats)
        pmat = reshape_pvalues(pvalues_corr)
        pmat_raw = reshape_pvalues(pvalues)
        matrix_export = {
            "statmap.joblib": statmat,
            "pmat.joblib": pmat,
            "pmat_raw.joblib": pmat_raw
        }
        save_run(parameters, joblib.dump, matrix_export, dirkey="statresults")

    resdf = pd.DataFrame({
        "edges": edges,
        "beta_AD": stats,
        "pvalues_raw": pvalues,
        "pvalues_fdr": pvalues_corr
    })
    exppath = save_run(
        parameters,
        lambda df, fname: df.to_csv(fname),
        {"resdf.csv": resdf, "tested_population.csv": df},
        dirkey="statresults"
    )

    print(exppath)
    
if __name__ == "__main__":
    main()
    