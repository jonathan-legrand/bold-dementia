"""
Functions for permutation testing in the context on Network Contigency Analysis
"""
from bold_dementia.utils.configuration import get_config
from bold_dementia.utils.saving import save_run
from bold_dementia.stats.univariate import make_fc_data, run_test, export_results, merge_configs
from bold_dementia.connectivity import reshape_pvalues, plot_matrix, Atlas
import random
import numpy as np
import pandas as pd

def generate_null(
    df:dict,
    edges:dict,
    parameters:dict,
    N:int=100,
    seed:int=1234
    ):
    random.seed(seed)
    idx_range = list(range(len(df)))
    permutation_scheme = [
        random.sample(idx_range, k=len(idx_range)) for _ in range(N)
    ]
    permuted_slopes = []

    for permutation in permutation_scheme:
        permuted_target = df.loc[permutation, "AD"].reset_index(drop=True)
        permuted_df = df.copy()
        permuted_df["AD"] = permuted_target
        results = run_test(permuted_df, edges, parameters)
        stats, pvalues = zip(*results)
        permuted_slopes.append(stats)

    permuted_slopes = pd.DataFrame(
        np.stack(permuted_slopes, axis=1), columns=[f"p_{i}" for i in range(N)]
    )
    permuted_slopes["edge"] = edges
    return permuted_slopes, permutation_scheme


    