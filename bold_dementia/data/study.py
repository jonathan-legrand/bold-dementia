from pandas import DataFrame

def make_control_idx(target: DataFrame, control: DataFrame) -> list:
    subset = control.sample(n=len(target), replace=False)
    return subset.index.to_list()
    