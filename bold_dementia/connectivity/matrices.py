import math
import numpy as np

from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from itertools import combinations
import math

def compute_mat_size(l):
    # Mat size is the positive root of :
    # n**2 - n - 2l = 0 
    # Where l is the length of pvalues array
    # and n is the square matrix size
    n = (1 + math.sqrt(1 + 8 * l)) / 2
    if n != int(n):
        raise ValueError(f"Array of lenght {l} cannot be reshaped as a square matrix")
    return int(n)
    

def vec_idx_to_mat_idx(l):
    n = compute_mat_size(l)
    return np.tril_indices(n, k=-1)
    

def reshape_pvalues(pvalues):
    l = len(pvalues)
    n = compute_mat_size(l)
    
    arr = np.zeros((n, n))
    pointer = 0
    for i in range(n):
        if i + pointer > pointer:
            arr[i, :i] = pvalues[pointer:pointer+i]
        pointer += i

    return arr + arr.T
    

def plot_matrix(
    mat, atlas, macro_labels=True, bounds=None, cmap="seismic"
):
    """Simplified version of the plot_matrices function. Only displays
    a single matrix.

    Args:
        mat (_type_): _description_
        atlas (Bunch): sklearn bunch containing labels and
        macro labels id macro_labels is True
        macro_labels (bool, optional): _description_. Defaults to True.
        bounds (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "seismic".

    """
    mat = mat.copy()
    n_regions = mat.shape[0]
    mat[list(range(n_regions)), list(range(n_regions))] = 0
    
    # In general we want a colormap that is symmetric around 0
    span = max(abs(mat.min()), abs(mat.max()))
    if bounds is None:
        bounds = (-span, span)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if macro_labels:
        networks = np.array(atlas.macro_labels)

        sort_index = np.argsort(networks)
        ticks = []
        lbls = []
        prev_label = None
        for i, label in enumerate(networks[sort_index]):
            if label != prev_label:
                ticks.append(i)
                lbls.append(label)
                prev_label = label
                ax.hlines(i, 0, n_regions, colors="black", linestyles="dotted")
                ax.vlines(i, 0, n_regions, colors="black", linestyles="dotted")

        ticks.append(i + 1)
        
    else:
        sort_index = np.arange(n_regions)
    
    sns.heatmap(
        mat[np.ix_(sort_index, sort_index)],
        ax=ax,
        vmin=bounds[0],
        vmax=bounds[1],
        cmap=cmap
    )

    if macro_labels:
        ax.yaxis.set_minor_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        ax.xaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        ax.set_yticklabels(lbls, rotation=0)
        ax.set_xticklabels(lbls, rotation=30)

    fig.tight_layout()
    return fig



def plot_matrices(
    cov, prec, title, labels, macro_labels=True, cov_bounds=(-1, 1), prec_bounds=None, cmap="seismic"
    ):
    """Plot covariance and precision matrices.
    For macro labels only schaeffer and rsn41 have been tested so far
    """
    prec = prec.copy()  # avoid side effects
    cov = cov.copy()

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    cov[list(range(size)), list(range(size))] = 0

    span = max(abs(prec.min()), abs(prec.max()))
    if prec_bounds is None:
        prec_bounds = (-span, span)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # We want network labels to span over several rows
    if macro_labels:
        networks = np.array(list(map(lambda x: str(x).split("_")[2], labels)))
        n_regions = len(labels)
        labels = None
        sort_index = np.argsort(networks)
        ticks = []
        lbls = []
        prev_label = None
        for i, label in enumerate(networks[sort_index]):
            if label != prev_label:
                ticks.append(i)
                lbls.append(label)
                prev_label = label
                ax1.hlines(i, 0, n_regions, colors="black", linestyles="dotted")
                ax2.hlines(i, 0, n_regions, colors="black", linestyles="dotted")
                ax1.vlines(i, 0, n_regions, colors="black", linestyles="dotted")
                ax2.vlines(i, 0, n_regions, colors="black", linestyles="dotted")

        ticks.append(i + 1)
        
    else:
        sort_index = np.arange(len(prec))

    sns.heatmap(cov[np.ix_(sort_index, sort_index)], ax=ax1, vmin=cov_bounds[0], vmax=cov_bounds[1], cmap=cmap)
    sns.heatmap(prec[np.ix_(sort_index, sort_index)], ax=ax2, cmap=cmap, vmin=prec_bounds[0], vmax=prec_bounds[1])

    if macro_labels:
        ax1.yaxis.set_minor_locator(FixedLocator(ticks))
        ax1.yaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        ax1.xaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        ax1.set_yticklabels(lbls, rotation=0)
        ax1.set_xticklabels(lbls, rotation=30)
    
    ax1.set_title("covariance")
    ax2.set_title("precision")
    fig.suptitle(title)
    fig.tight_layout()

    return fig, ax1, ax2

def network_to_network_connectivity(matrix, network_to_idx):
    """

    Args:
        matrix (_type_): Matrix should have the block structure 
        described in network_to_idx!
        network_to_idx (_type_): _description_

    Yields:
        _type_: _description_
    """
    for network_a, network_b in combinations(network_to_idx.index, 2):
        loc_a, loc_b = network_to_idx[network_a], network_to_idx[network_b]
        # FIXME This can't be right
        connectivity = matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]].mean()
        yield network_a, network_b, connectivity

def plot_ordered_matrix(
    mat, atlas, bounds=None, cmap="seismic",
):
    """Simplified version of the plot_matrices function. Only displays
    a single matrix.

    Args:
        mat (_type_): _description_
        atlas (Bunch): sklearn bunch containing labels and
        macro labels id macro_labels is True
        macro_labels (bool, optional): _description_. Defaults to True.
        bounds (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "seismic".

    """
    mat = mat.copy()
    n_regions = mat.shape[0]
    mat[list(range(n_regions)), list(range(n_regions))] = 0
    
    # In general we want a colormap that is symmetric around 0
    span = max(abs(mat.min()), abs(mat.max()))
    if bounds is None:
        bounds = (-span, span)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.heatmap(
        mat,
        ax=ax,
        vmin=bounds[0],
        vmax=bounds[1],
        cmap=cmap,
        xticklabels=atlas.labels,
        yticklabels=atlas.labels
    )


    fig.tight_layout()
    return fig


def group_by_networks(macro_labels):
    networks = np.array(macro_labels)
    sort_index = np.argsort(networks)

    ticks = []
    lbls = []
    prev_label = None
    for i, label in enumerate(networks[sort_index]):
        if label != prev_label:
            print(label)
            ticks.append(i)
            lbls.append(label)
            prev_label = label

    ticks.append(i+1)
    return ticks, sort_index