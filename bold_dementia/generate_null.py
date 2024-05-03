import sys
from pathlib import Path
import argparse
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
from bold_dementia.stats.permutations import generate_null, generate_null_dask
from bold_dementia.stats.univariate import make_fc_data, merge_configs

config = get_config()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate empirical null distribution with permutations")
    parser.add_argument(
        "maps_name",
        help="Name of maps dir in the connectivity directory"
    )
    parser.add_argument(
        "model_spec_path",
        help="Path to the yaml specification of the linear model to be fit on edges"
    )

    parser.add_argument(
        "--n_permutations",
        help="Number of point in the null distribution",
        type=int,
        default=5000
    )
    
    parser.add_argument(
        "--use_dask",
        help="Set to True to use dask to submit jobs on SLURM",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--n_jobs",
        help="Number of parallel processes for fitting on permuted data",
        type=int,
        default=10
    )
    
    parser.add_argument(
        "--seed",
        help="Custom seed for scan selection when there are multiple scans per subject",
        type=int,
        default=config["seed"]
    )
    return parser


def generate_and_export(
    maps_name,
    model_specs_path,
    n_permutations,
    use_dask,
    n_jobs,
    seed,
    ):
    maps_path = Path(config["output_dir"]) / "connectivity" / maps_name
    
    maps_specs = get_config(maps_path / "parameters.yml")
    model_specs = get_config(model_specs_path)
    print(model_specs)

    df, edges, parameters = make_fc_data(maps_path, maps_specs, model_specs, seed=seed)
    
    # Experimental dask version
    if use_dask:
        with SLURMCluster(
            cores=1,
            memory="1GB",
            walltime="00:05:00",
            log_directory="/tmp"
        ) as cluster:
            cluster.scale(n_jobs)
            client = Client(cluster)
            print(client.dashboard_link)
            permuted_slopes, permutation_scheme = generate_null_dask(
                df, edges, parameters, client, N=n_permutations
            )
    else:
        permuted_slopes, _ = generate_null(
            df, edges, parameters, N=n_permutations, seed=seed, n_jobs=n_jobs
        )
    params = merge_configs(maps_specs, model_specs)
    export_path = save_run(
        params,
        lambda obj, path: obj.to_csv(path),
        {f"null_distribution_{n_permutations}.csv": permuted_slopes,},
        dirkey="statresults"
    )
    print(f"Saved results in {export_path}")


if __name__ == "__main__":
    
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    generate_and_export(*vars(args).values())