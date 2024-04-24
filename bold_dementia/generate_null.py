import sys
from pathlib import Path
import warnings
from bold_dementia import get_config

import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
                       

from bold_dementia.models.matrix_GLM import fit_edges
from bold_dementia.data.volumes import add_volumes
from utils.saving import save_run
from utils.iterables import join, all_connectivities
from bold_dementia.stats.permutations import generate_null
from bold_dementia.stats.univariate import make_fc_data, merge_configs

config = get_config()



def main():
    maps_path = Path(config["connectivity_matrices"]) / sys.argv[1]
    model_specs_path = Path(sys.argv[2])
    
    maps_specs = get_config(maps_path / "parameters.yml")
    model_specs = get_config(model_specs_path)
    print(model_specs)

    df, edges, parameters = make_fc_data(maps_path, maps_specs, model_specs)
    
    #cluster = SLURMCluster(
    #    cores=4,
    #    processes=4,
    #    memory="20GB",
    #    job_mem='500MB',
    #    walltime="03:00:00",
    #)
    #client = cluster.get_client()
    #cluster.scale(jobs=10)
    #print(client.dashboard_link)
    #with joblib.parallel_backend('dask', scatter=[df]):
    permuted_slopes, _ = generate_null(
        df, edges, parameters, N=1000, seed=config["seed"]
    )
    params = merge_configs(maps_specs, model_specs)
    export_path = save_run(
        params,
        lambda obj, path: obj.to_csv(path),
        {"null_distribution.csv": permuted_slopes,},
        dirkey="statresults"
    )
    print(f"Saved results in {export_path}")


if __name__ == "__main__":
    main()