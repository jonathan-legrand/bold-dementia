import warnings
from pandas import DataFrame
import pandas as pd
from nilearn import signal
from nilearn.interfaces.fmriprep import load_confounds

def make_control_idx(target: DataFrame, control: DataFrame) -> list:
    subset = control.sample(n=len(target), replace=False)
    return subset.index.to_list()

def p(s: pd.Series)->float:
    """

    Args:
        s (pd.Series): Series with two categories

    Returns:
        float: Proportion of most frequent variable
    """
    # Hard coded hack because sex is the only thing we need
    prop = s.value_counts()["Féminin"] / len(s)
    return "Féminin", prop


# TODO Abstract further, the two following functions are doing
# the same thing
def balance_control_cat(
    pos:DataFrame,
    control:DataFrame,
    col_name:str,
    tol:float=0.1,
):
    def gap_func(pos, control):
        value, target_p = p(pos[col_name])
        _, control_p = p(control[col_name])
        return target_p - control_p, value

    gap, value = gap_func(pos, control)
    counter = 0
    while abs(gap) > tol:
        counter += 1

        if gap > 0:
            mask = (control[col_name] != value)
        else:
            mask = (control[col_name] == value)
        
        idx_to_drop = control[mask].sample(n=1, random_state=1234).index[0]
        print(control.loc[idx_to_drop, col_name], end=", new gap = ")
            
        control = control.drop(idx_to_drop)
        
        if len(control) <= len(pos):
            raise ValueError("Removed too many subjects from control")
        
        gap, value = gap_func(pos, control)

        print(gap, end=", ")
        print(f"{len(control)} controls left")

    return pos, control 

# Turn into dispatcher
def balance_control(
    pos:DataFrame,
    control:DataFrame,
    col_name:str,
    tol:float=1,
):
    """Balance control with pos on the quantitative variable named col_name

    Args:
        pos (DataFrame): _description_
        control (DataFrame): _description_
        col_name (str): _description_
        tol (float, optional): _description_. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    gap = pos[col_name].mean() - control[col_name].mean()
    # Usually the age is lower in control group
    counter = 0
    while abs(gap) > tol:
        counter += 1
        print(f"#{counter}, removed {col_name} = ", end=" ")
        
        if gap > 0:
            idx_to_drop = control[col_name].idxmin()
        else:
            idx_to_drop = control[col_name].idxmax()
            
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
        print(f"Processing {fpath}")
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
            