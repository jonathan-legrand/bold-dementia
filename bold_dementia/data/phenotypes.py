
from datetime import datetime
import pandas as pd

def days_to_onset(
    reference: pd.Series, demence_dat: pd.Series
) -> pd.Series:
    timedelta = demence_dat - reference
    return timedelta.map(lambda x: x.days)
    