"""
Functions for permutation testing in the context on Network Contigency Analysis
"""
from bold_dementia.utils.configuration import get_config
from bold_dementia.utils.saving import save_run
from bold_dementia.stats.univariate import make_fc_data, run_test_serial, export_results, merge_configs
from bold_dementia.connectivity import reshape_pvalues, plot_matrix, Atlas
import random
import numpy as np
import pandas as pd
import joblib

from dask.distributed import LocalCluster

import dask.delayed as delayed
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster

import dask.array as da

import dask.delayed as delayed 

from dask.dataframe import from_pandas

def generate_null(
    df:dict,
    edges:dict,
    parameters:dict,
    N:int=100,
    seed:int=1234,
    n_jobs=8
    ):

    random.seed(seed)
    
    idx_range = list(range(len(df)))
    permutation_scheme = [
        random.sample(idx_range, k=len(idx_range)) for _ in range(N)
    ]

    def single_call(permutation):
        permuted_target = df.loc[permutation, "AD"].reset_index(drop=True)
        permuted_df = df.copy()
        permuted_df["AD"] = permuted_target
        results = run_test_serial(permuted_df, edges, parameters)
        stats, pvalues = zip(*results)
        return stats
        
    parallel = joblib.Parallel(verbose=2, n_jobs=n_jobs)
    calls = (joblib.delayed(single_call)(permutation) for permutation in permutation_scheme)
    permuted_slopes = parallel(calls)

    permuted_slopes = pd.DataFrame(
        np.stack(permuted_slopes, axis=1), columns=[f"p_{i}" for i in range(N)]
    )
    permuted_slopes["edge"] = edges
    return permuted_slopes, permutation_scheme

from dask.distributed import Client, progress

def generate_null_dask(
    df:dict,
    edges:dict,
    parameters:dict,
    client:Client,
    N:int=100,
    seed:int=1234,
    ):
    
    random.seed(seed)
    
    idx_range = list(range(len(df)))
    permutation_scheme = [
        random.sample(idx_range, k=len(idx_range)) for _ in range(N)
    ]

    def single_call(permutation):
        import sys
        sys.path.append("/homes_unix/jlegrand/AD-prediction")
        from bold_dementia.stats.univariate import run_test_serial

        permuted_target = df.loc[permutation, "AD"].reset_index(drop=True)
        permuted_df = df.copy()
        permuted_df["AD"] = permuted_target
        results = run_test_serial(permuted_df, edges, parameters)
        stats, pvalues = zip(*results)
        return stats
    
    print(client)

    futures = client.map(single_call, permutation_scheme)
    progress(futures, notebook=False)
    permuted_slopes = [future.result() for future in futures]

    permuted_slopes = pd.DataFrame(
        np.stack(permuted_slopes, axis=1), columns=[f"p_{i}" for i in range(N)]
    )
    permuted_slopes["edge"] = edges
    return permuted_slopes, permutation_scheme


    