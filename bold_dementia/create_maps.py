import warnings
from pathlib import Path
import os
import json
import joblib
from typing import Callable

import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.linalg as npl
import math
import random
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

from sklearn.utils import Bunch
from sklearn import covariance
from nilearn.connectome import ConnectivityMeasure

from nilearn import plotting

from bold_dementia.data.study import balance_control, balance_control_cat, load_signals
from bold_dementia.data.memento import Memento, MementoTS, past_diag_AD, healthy_control
from bold_dementia.connectivity.atlases import Atlas
from bold_dementia.connectivity.matrices import plot_matrices, reshape_pvalues
from bold_dementia import get_config, get_custom_config


# TODO Add possibility to override default conf
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


def save_run(run_config: str, save_func: Callable, save_mapping: dict) -> Path:
    """Save current run object and parameters

    Args:
        save_func (Callable): dedicated func which takes (obj, path) 
        as paramaters, to save the python objects in save mapping
        save_mapping (dict): Map between fname and python object

    Returns:
        Path: path of the folder containing all the saved objects
    """

    name = f"atlas-{run_config['ATLAS']}_{run_config['NAME']}"

    experience_path = Path(config["connectivity_matrices"]) / name

    if not os.path.exists(experience_path):
        os.makedirs(experience_path)

    with open(experience_path / "parameters.json", "w") as stream:
        json.dump(run_config, stream)

    for fname, obj in save_mapping.items():
        save_func(obj, experience_path / fname)

    return experience_path

def create_maps(run_config):
    atlas = Atlas.from_name(
        run_config["ATLAS"],
        run_config["SOFT"]
    )
    cache_dir = Path(config["bids_dir"]) / "derivatives" / f"{atlas.name}"
    memento = MementoTS(cache_dir=cache_dir, target_func=lambda row: row)

    with warnings.catch_warnings(category=FutureWarning, action="ignore"):
        AD_signals_ub, control_signals_ub, pm, nm = load_signals(
            memento,
            past_diag_AD,
            healthy_control,
            clean_signal=run_config["CLEAN_SIGNAL"],
            confounds_strategy=run_config["confounds_strategy"]
        )

    if run_config["BALANCE_STRAT"] != []:
        raise NotImplementedError(
            "This is left as an exercise to the reader. \
                Alternatively use the notebook.")
    else:
        balanced_AD, balanced_meta = pm, nm

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

    joblib_export = {
        "AD.joblib": gcov.covariances_[AD_indices, :, :],
        "control.joblib": gcov.covariances_[control_indices, :, :],
        "AD_series_ub.joblib": AD_signals_ub,
        "control_series_ub.joblib": control_signals_ub
    }
    csv_export = {
        "balanced_AD.csv": balanced_AD,
        "balanced_control.csv": balanced_meta,
        "AD_series_ub.csv": pm,
        "control_series_ub.csv": nm,
    }

    save_run(run_config, joblib.dump, joblib_export)
    save_run(run_config, lambda df, fname: df.to_csv(fname), csv_export)

import sys

if __name__ == "__main__":
    try:
        run_config = get_custom_config(sys.argv[1])
        print("Loaded custom config :")
    except IndexError:
        run_config = config["default_run"]
        print("No config path provided, using default :")
    print(run_config)
    
    create_maps(run_config)



    


