import warnings
from pandas import DataFrame
import pandas as pd
from nilearn import signal
from nilearn.interfaces.fmriprep import load_confounds

def make_control_idx(target: DataFrame, control: DataFrame) -> list:
    subset = control.sample(n=len(target), replace=False)
    return subset.index.to_list()
    

def balance_control(pos, control, col_name, tol=1):
    gap = pos[col_name].mean() - control[col_name].mean()
    # Usually the age is lower in control group
    counter = 0
    while gap > tol:
        counter += 1
        print(f"#{counter}, removed age = ", end=" ")
        idx_to_drop = control[col_name].idxmin()
        print(control.loc[idx_to_drop, col_name], end=", new gap = ")
        control = control.drop(idx_to_drop)
        

        if len(control) <= len(pos):
            raise ValueError("Removed too many subjects from control")
        gap = pos[col_name].mean() - control[col_name].mean()
        print(gap, end=", ")
        print(f"{len(control)} controls left")

    return pos, control
        

def load_signals(dataset, is_pos_func, is_neg_func, clean_signal=False, confounds_strategy=None):
    pos_ts = []
    neg_ts = []
    pos_meta = []
    neg_meta = []
    for ts, row, fpath in iter(dataset):
        if clean_signal:
            confounds, sample_mask = load_confounds(
                fpath, **confounds_strategy
            )
            with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
                cleaned_ts = signal.clean(
                    ts,
                    sample_mask=sample_mask,
                    confounds=confounds,
                    standardize="zscore_sample"
                )
        else:
            cleaned_ts = ts
        if is_pos_func(row):
            pos_ts.append(cleaned_ts)
            pos_meta.append(row)
        elif is_neg_func(row):
            neg_ts.append(cleaned_ts)
            neg_meta.append(row)

    pos_meta = pd.DataFrame(pos_meta).reset_index(drop=True)
    neg_meta = pd.DataFrame(neg_meta).reset_index(drop=True)
    return pos_ts, neg_ts, pos_meta, neg_meta
            