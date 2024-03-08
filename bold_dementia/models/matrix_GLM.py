from formulaic import model_matrix
import statsmodels.api as sm
import numpy as np


def fit_edges(edge_name, dataframe, rhs_formula, groups):
    lhs, rhs = model_matrix(f"`{edge_name}` ~ {rhs_formula}", dataframe)
    if groups is None:
        model = sm.OLS(endog=lhs, exog=rhs)
    else:
        model = sm.MixedLM(endog=lhs, exog=rhs, groups=dataframe[groups])
    result = model.fit() # TODO Explore method influence
    return result.params.AD, result.pvalues.AD