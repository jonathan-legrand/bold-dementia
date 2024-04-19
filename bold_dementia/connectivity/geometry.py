from nilearn.connectome.connectivity_matrices import _geometric_mean, _map_eigenvalues
from joblib import Parallel, delayed
import numpy as np


def project_to_tangent(matrices):
    """
    Following Ng et al. 2016, we use :
    Log_B(A) = B^(1/2) * log_m(B^(-1/2) * A * B^(-1/2)) * B^(1/2)
    with B the reference point and A the matrix to be projected.
    Nilearn does not multiply by B, weird# TODO Try with B projection
    """

    print("Computing geometrical mean...")
    B = _geometric_mean(matrices, max_iter=5, tol=1e-2)
    print("Computing whitening...")
    W = _map_eigenvalues(
        lambda d: 1/np.sqrt(d), B
    )
    def project(A, W):
        return _map_eigenvalues(
            np.log, W @ A @ W
        )
    
    print("Projecting matrices...")
    projected_matrices = []
    for A in matrices:
        projected = project(A, W)
        projected_matrices.append(projected)
    print("Done")
    return np.stack(projected_matrices, axis=0)
        