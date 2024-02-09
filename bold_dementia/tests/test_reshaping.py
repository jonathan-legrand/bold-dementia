import pytest
import numpy as np
from bold_dementia.connectivity.matrices import reshape_pvalues

@pytest.fixture
def rng():
    return np.random.default_rng(seed=1234)
    
@pytest.fixture
def pvalues(rng):
    n = rng.integers(5, 500)
    l = int((n  * (n-1)) / 2)
    pvalues = rng.uniform(0, 1, size=(l,))
    return pvalues

def f(l):
    return (1 + np.sqrt(1 + 8 * l)) / 2

@pytest.fixture
def non_squarable(rng:np.random.Generator)->np.ndarray:
    """
    Args:
        rng (np.ndarray): rng from fixture

    Returns:
        np.ndarray: non squarable pvalues
    """
    x = np.arange(50)
    # Choose l sizes such that the root of n**2 - n - 2l = 0
    # is not an integer, so the square matrix cannot be
    # reconstructed
    non_squarable = [i for i in range(len(f(x))) if f(x)[i] != int(f(x)[i])]
    size = rng.choice(non_squarable)
    return rng.uniform(0, 1, (size,))
    
def test_reshape_random(pvalues):
    matrix = reshape_pvalues(pvalues)
    assert np.equal(matrix, matrix.T).all(), "Output matrix is not symmetric"

def test_nonsquarable(non_squarable):
    with pytest.raises(ValueError):
        reshape_pvalues(non_squarable)