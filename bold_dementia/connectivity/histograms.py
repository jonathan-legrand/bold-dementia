import matplotlib.pyplot as plt
import numpy as np

def fast_hist(matrix:np.ndarray, ax=None, **plot_kwargs):
    """Plot values of arrays containing
    individuals correlations

    Args:
        matrix (np.ndarray): (n_subjects, n_regions, r_regions)

    """
    n_regions = matrix.shape[1]
    n_subjects = matrix.shape[0]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    # Passing the array is slower
    for i in range(n_subjects):
        tst = matrix[i, :, :].reshape((n_regions ** 2))
        ax.hist(tst, histtype="step", **plot_kwargs)
    ax.set_xlim(-1, 1)
    return fig, ax