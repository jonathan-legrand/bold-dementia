
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
from bold_dementia.stats.univariate import make_fc_data, run_test, export_results

config = get_config()


import os
#os.environ['PYTHONWARNINGS']='ignore::ConvergenceWarning'

def main():
    maps_path = Path(config["output_dir"]) / "connectivity" / sys.argv[1]
    model_specs_path = Path(sys.argv[2])
    
    maps_specs = get_config(maps_path / "parameters.yml")
    model_specs = get_config(model_specs_path)
    print(model_specs)

    df, edges, parameters = make_fc_data(maps_path, maps_specs, model_specs)
    print(df.head())

    results = run_test(df, edges, parameters) # TODO Chain functions
    export_results(results, edges, maps_specs, model_specs)
    
    
if __name__ == "__main__":
    main()
    