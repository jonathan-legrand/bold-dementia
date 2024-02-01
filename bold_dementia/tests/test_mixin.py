from bold_dementia.models.transformers import ListMixin
from sklearn.preprocessing import StandardScaler
import pytest
import numpy as np

class ListScaler(StandardScaler, ListMixin):
    pass

@pytest.fixture
def X_unscaled():
    n = 10
    rng = np.random.default_rng(seed=1234)
    X = []
    for _ in range(n):
        mean = rng.random() * 10
        sd = rng.exponential(10)
        size = (rng.integers(1, 20), rng.integers(1, 10))
        X.append(rng.normal(mean, sd, size=size))
    return X
        

def test_list_mixin(X_unscaled: list):
    scaler = ListScaler()
    X_scaled = scaler.fit_transform_lst(X_unscaled)

    is_centered = lambda arr: np.allclose(arr.mean(axis=0), 0)
    is_std = lambda arr: np.allclose(arr.std(axis=0), 1)

    are_centered = map(is_centered, X_scaled)
    are_std = map(is_std, X_scaled)
    
    assert all(are_centered) & all(are_std)
