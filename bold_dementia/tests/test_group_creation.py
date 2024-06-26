import pytest
import numpy as np
import pandas as pd
from bold_dementia import get_config
from bold_dementia.data.study import balance_control, balance_control_cat
config = get_config()

@pytest.fixture
def rng():
    return np.random.default_rng(seed=1234)

def make_df(expected_age, scale, p_female, size, rng):
    age = rng.normal(loc=expected_age, scale=scale, size=size)
    sex = rng.choice(
        ("Féminin", "Masculin"),
        size=size,
        p=(p_female, 1-p_female)
    )
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex
        }
    )
def female_prop(s):
    return (s.value_counts()["Féminin"] / len(s))

def test_younger_control(rng):
    control = make_df(60, scale=10, p_female=0.70, size=1000, rng=rng)
    AD = make_df(70, scale=10, p_female=0.50, size=100, rng=rng)
    _, new_control = balance_control(AD, control, "age", tol=config["age_tol"])
    
    assert abs(new_control.age.mean() - AD.age.mean()) < config["age_tol"]

def test_older_control(rng):
    control = make_df(80, scale=10, p_female=0.70, size=1000, rng=rng)
    AD = make_df(70, scale=10, p_female=0.50, size=100, rng=rng)
    _, new_control = balance_control(AD, control, "age", tol=config["age_tol"])
    
    assert abs(new_control.age.mean() - AD.age.mean()) < config["age_tol"] 

def test_remove_females(rng):
    control = make_df(80, scale=10, p_female=0.70, size=1000, rng=rng)
    AD = make_df(70, scale=10, p_female=0.50, size=100, rng=rng)

    _, new_control = balance_control_cat(AD, control, "sex", tol=config["sex_prop_tol"])

    assert abs(female_prop(new_control.sex) - female_prop(AD.sex)) < config["sex_prop_tol"]

def test_remove_males(rng):
    control = make_df(80, scale=10, p_female=0.30, size=1000, rng=rng)
    AD = make_df(70, scale=10, p_female=0.60, size=100, rng=rng)

    _, new_control = balance_control_cat(AD, control, "sex", tol=config["sex_prop_tol"])

    assert abs(female_prop(new_control.sex) - female_prop(AD.sex)) < config["sex_prop_tol"]