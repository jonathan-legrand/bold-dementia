from formulaic import model_matrix
import statsmodels.api as sm
import numpy as np


def fit_edges(edge_name, dataframe):
    lhs, rhs = model_matrix(f"`{edge_name}` ~ AD + scale(current_scan_age) + SEX + NIVETUD", dataframe)
    model = sm.MixedLM(endog=lhs, exog=rhs, groups=dataframe["sub"])
    try:
        result = model.fit(method="bfgs") # TODO Explore method influence
        return result.params.AD, result.pvalues.AD
    except np.linalg.LinAlgError:
        return np.nan, np.nan