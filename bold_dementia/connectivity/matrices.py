import matplotlib.pyplot as plt
from nilearn import plotting
import math
import numpy as np

def reshape_pvalues(pvalues):
    l = len(pvalues)
    
    # Mat size is the positive root of :
    # n**2 - n - 2l = 0 
    # Where l is the length of pvalues array
    # and n is the square matrix size
    n = (1 + math.sqrt(1 + 8 * l)) / 2
    if n != int(n):
        raise ValueError(f"Array of lenght {l} cannot be reshaped as a square matrix")
    n = int(n)
    
    arr = np.zeros((n, n))
    pointer = 0
    for i in range(n):
        if i + pointer > pointer:
            arr[i, :i] = pvalues[pointer:pointer+i]
        pointer += i

    return arr + arr.T
    

def plot_matrices(cov, prec, title, labels):
    """Plot covariance and precision matrices, for a given processing."""
    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    #span = max(abs(prec.min()), abs(prec.max()))
    span = 1

    # Display covariance matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    plotting.plot_matrix(
        cov,
        cmap=plotting.cm.bwr,
        vmin=-1,
        vmax=1,
        title=f"{title} / covariance",
        labels=labels,
        axes=ax1
    )
    # Display precision matrix
    plotting.plot_matrix(
        prec,
        cmap=plotting.cm.bwr,
        vmin=-span,
        vmax=span,
        title=f"{title} / precision",
        labels=labels,
        axes=ax2
    )

    return fig