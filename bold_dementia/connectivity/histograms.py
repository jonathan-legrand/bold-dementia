import matplotlib.pyplot as plt
import numpy as np

def fast_hist(matrix:np.ndarray, **plot_kwargs):
    """Plot values of arrays containing
    individuals correlations

    Args:
        matrix (np.ndarray): (n_subjects, n_regions, r_regions)

    """
    n_regions = matrix.shape[1]
    n_subjects = matrix.shape[0]
    fig, ax = plt.subplots(1, 1)

    # Passing the array is slower
    for i in range(n_subjects):
        tst = matrix[i, :, :].reshape((n_regions ** 2))
        ax.hist(tst, histtype="step", **plot_kwargs)
    ax.set_xlim(-1, 1)
    return fig, ax