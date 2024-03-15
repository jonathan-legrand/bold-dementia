

from itertools import product
import numpy as np
import numpy.linalg as npl
from nilearn.connectome import cov_to_corr


def generate_topology(target_label, labels):
    n_regions = len(labels)
    labels = np.array(labels)

    mat = np.zeros((n_regions, n_regions))
    network_idx = np.where(labels == target_label)[0]
    mat_coords = tuple(product(network_idx, network_idx))
    for coords in mat_coords:
        mat[coords] = 1
    return mat
    

rng = np.random.default_rng()
import matplotlib.pyplot as plt


def generate_correlations(n_subjects, topology, loc, scale, snr=2, from_prec=False, debug_display=False):
    correlations = []
    mask = topology > 0
    n_regions = len(topology)
    if debug_display:
        plt.hist(rng.normal(loc=loc, scale=scale, size=n_regions * 10))
        plt.title(f"Source normal distribution N({loc}, {scale})")
        plt.show()

    for _ in range(n_subjects):
        synthetic = topology.copy()
        synthetic[mask] = rng.normal(loc=loc, scale=scale, size=(mask.sum())) # TODO More realistic distribution?
        synthetic += rng.normal(loc=0, scale=np.sqrt(loc**2/snr), size=synthetic.shape) # Assuming SNR = (mu ^ 2) / (sigma ^ 2)

        synthetic += np.eye(n_regions)
        synthetic = synthetic.T @ synthetic
        if debug_display:
            plt.hist(synthetic[mask], histtype="step", color="blue")

        np.testing.assert_almost_equal(synthetic, synthetic.T)
        eigenvalues = np.linalg.eigvalsh(synthetic)
        if eigenvalues.min() < 0:
            raise ValueError(
                "Failed generating a positive definite precision "
                "matrix. Decreasing n_features can help solving "
                "this problem."
            )

        if from_prec:
            covariance = npl.inv(synthetic)
        else:
            covariance = synthetic
            
        correlation = cov_to_corr(covariance)

        if debug_display:
            plt.hist(covariance[mask], histtype="step", color="green")
            plt.hist(correlation[mask], histtype="step", color="red")

        correlations.append(correlation) # We could yield but what's the point
    if debug_display:
        plt.title("Random values in topology mask")
        plt.show()
    return correlations