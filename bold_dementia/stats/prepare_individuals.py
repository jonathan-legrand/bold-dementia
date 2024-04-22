
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
from bold_dementia.utils.saving import save_run
from bold_dementia.utils.iterables import join, all_connectivities

config = get_config("/bigdata/jlegrand/AD-prediction/config.yml") #Awful

def make_fc_data(maps_path, maps_spec, model_spec):
    
    pos = maps_spec["posfunc"]
    neg = maps_spec["negfunc"]
    AD_matrices = joblib.load(maps_path / f"{pos}.joblib")
    control_matrices = joblib.load(maps_path / f"{neg}.joblib")
    atlas = Atlas.from_name(maps_spec["ATLAS"], maps_spec["SOFT"])

    AD_df = pd.read_csv(maps_path / f"balanced_{pos}.csv", index_col=0)
    control_df = pd.read_csv(maps_path / f"balanced_{neg}.csv", index_col=0)
    df = pd.concat((AD_df, control_df))

    # Whether to perform analysis on a network level to tame FDR
    if "BY_BLOCK" in model_spec.keys() and model_spec["BY_BLOCK"] is True:
        print("BY_BLOCK is True, grouping regions into networks...")
        AD_matrices, _ = group_groupby(AD_matrices, atlas)
        control_matrices, labels = group_groupby(control_matrices, atlas)

        AD_fc = pd.concat(edge_format(block, labels) for block in AD_matrices).reset_index(drop=True)
        control_fc = pd.concat(edge_format(block, labels) for block in control_matrices).reset_index(drop=True)
        edges = AD_fc.columns.to_list()
        
        fc = pd.concat((AD_fc, control_fc)).reset_index(drop=True)
        #fc = fc.apply(np.arctanh) TODO UNCOMMENT ME PLEASE
        
        # k is named that way because of numpy
        k:int = 0
        
        print("New labels : ", end="")
        print(labels)

    else:
        labels = atlas.labels
        k:int = -1
        AD_vec = np.array([z_transform_to_vec(mat, k) for mat in AD_matrices])
        control_vec = np.array([z_transform_to_vec(mat, k) for mat in control_matrices])

        fc = np.vstack((AD_vec, control_vec))
        l = fc.shape[1]
        rows, cols = vec_idx_to_mat_idx(l)
        edges = [f"{labels[i]}_{labels[j]}" for i, j in zip(rows, cols)]
        fc = pd.DataFrame(fc, columns=edges)

    
    print(fc.head())
    df["AD"] = np.where(df.scan_to_onset < 0, 1, 0) # This is not flexible enough
    df = pd.concat([df.reset_index(drop=True), fc], axis=1, join="inner")
    df = df.drop(df[df.MA == 0].index)

    # If the model does not account for subjects, then they should be unique
    if model_spec["GROUPS"] != "sub": 
        try:
            df = df.groupby("sub").sample(n=1, random_state=eval(sys.argv[3]))
        except IndexError:
            df = df.groupby("sub").sample(n=1, random_state=1234)
    else:
        print("Using several scans per subect and mixed models")

    # TODO Pass TIV only to be more efficient?
    if "total intracranial" in model_spec["RHS_FORMULA"]:
        print("Add intracranial volumes to phenotypes")
        df = add_volumes(df, config["volumes"])
    
    return df, edges, model_spec