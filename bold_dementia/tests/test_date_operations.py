"""
Never ever trust dates
"""
from datetime import datetime
from bold_dementia.data.phenotypes import days_to_onset
from bold_dementia.data.memento import Memento


import pandas as pd
import pytest

@pytest.fixture
def phenotypes():
    return Memento.load_phenotypes(
        "/bigdata/jlegrand/data/Memento/phenotypes.tsv",
        augmented=False
    )

@pytest.fixture
def left_censored(phenotypes):
    return phenotypes[phenotypes.INCCONSDAT_D >= phenotypes.DEMENCE_DAT]

@pytest.fixture
def right_censored(phenotypes):
    return phenotypes[phenotypes.DEMENCE_DAT.isna()]

def test_right_censoring(right_censored):
    for tstep in {"INCCONSDAT_D", "M000", "M024", "M048"}:
        days = days_to_onset(right_censored.loc[:, tstep], right_censored.DEMENCE_DAT)
        assert days.isna().all() 

def test_left_censoring(left_censored):
    days = days_to_onset(left_censored.M000, left_censored.DEMENCE_DAT)
    assert (days <= 0).all()

