import warnings
from pathlib import Path
import os
import json
import joblib

import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.linalg as npl
import math
from scipy.io import savemat
from statsmodels.stats.multitest import fdrcorrection

from sklearn.utils import Bunch
from sklearn import covariance
from nilearn.connectome import ConnectivityMeasure

from nilearn import plotting

from bold_dementia.data.study import balance_control, balance_control_cat, load_signals
from bold_dementia.data.memento import (
    Memento, MementoTS, past_diag_AD, converter, all_subs, delete_me
)
from bold_dementia.connectivity.atlases import Atlas
from bold_dementia.connectivity.matrices import plot_matrices, reshape_pvalues
from bold_dementia import get_config
from bold_dementia.utils.saving import save_run

# TODO Merge with normal create_maps

config = get_config()

def compute_cov_prec(time_series):

    pipe = ConnectivityMeasure(
        covariance.LedoitWolf(),
        kind="covariance"
    )
    
    c = pipe.fit_transform(time_series)
    return Bunch(
        covariances_=c,
        precisions_=npl.inv(c) # I don't like this
    )

def create_maps(run_config):
    atlas = Atlas.from_name(
        run_config["ATLAS"],
        run_config["SOFT"]
    )
    try:
        cache_dir = run_config["cache_dir"]
    except KeyError:
        cache_dir = Path(config["bids_dir"]) / "derivatives" / f"{atlas.name}"
    print(f"Fetching time series in {cache_dir}")
    memento = MementoTS(cache_dir=cache_dir, target_func=lambda row: row)
    posfunc_name = run_config["posfunc"] if "posfunc" in run_config.keys() else "converter"
    negfunc_name = run_config["negfunc"] if "negfunc" in run_config.keys() else "control"
    print(f"Selecting with {posfunc_name} and {negfunc_name}")
    with warnings.catch_warnings(category=FutureWarning, action="ignore"):
        AD_signals_ub, control_signals_ub, pm, nm = load_signals(
            memento,
            eval(posfunc_name),
            eval(negfunc_name),
            clean_signal=run_config["CLEAN_SIGNAL"],
            confounds_strategy=run_config["confounds_strategy"]
        )

    if "complete_subs_only" in run_config["BALANCE_STRAT"]:

        def complete_subs(df):
            counts = df.groupby("sub")["ses"].count()
            complete_subs = counts[(counts == 3)].index.to_list()
            return df[df["sub"].isin(complete_subs)]

        #balanced_AD = complete_subs(pm)
        balanced_AD = complete_subs(pm)
        balanced_meta = complete_subs(nm)
    elif run_config["BALANCE_STRAT"] == []:
        balanced_AD, balanced_meta = pm, nm
    else:
        raise NotImplementedError(
            "This is left as an exercise to the reader. \
                Alternatively use the notebook.")

    balanced_signals = [control_signals_ub[idx] for idx in balanced_meta.index]
    AD_signals = [AD_signals_ub[idx] for idx in balanced_AD.index]

    time_series = AD_signals + balanced_signals
    AD_indices = list(range(len(AD_signals)))
    control_indices = list(range(len(AD_signals), len(time_series)))

    n = len(time_series)
    print(f"Study on {n} subjects")
    
    print("Computing covariance", end="... ")
    gcov = compute_cov_prec(time_series)
    print("Finished, exporting results")

    mat_export = {
        f"{posfunc_name}.mat": gcov.covariances_[AD_indices, :, :],
        f"{negfunc_name}.mat": gcov.covariances_[control_indices, :, :],
        f"{posfunc_name}_series_ub.mat": AD_signals_ub,
        f"{negfunc_name}_series_ub.mat": control_signals_ub
    }
    csv_export = {
        f"balanced_{posfunc_name}.csv": balanced_AD,
        f"balanced_{negfunc_name}.csv": balanced_meta,
        f"{posfunc_name}_series_ub.csv": pm,
        f"{negfunc_name}_series_ub.csv": nm,
    }

    def to_matlab(obj, path):
        savemat(path, {"matrices": obj}, appendmat=False)

    save_run(run_config, to_matlab, mat_export)
    exppath = save_run(run_config, lambda df, fname: df.to_csv(fname), csv_export)
    return exppath

import sys

if __name__ == "__main__":
    try:
        run_config = get_config(sys.argv[1])
        print("Loaded custom config :")
    except IndexError:
        run_config = config["default_run"]
        print("No config path provided, using default :")
    print(run_config)
    
    p = create_maps(run_config)
    print(f"Saved output in {p}")



    


