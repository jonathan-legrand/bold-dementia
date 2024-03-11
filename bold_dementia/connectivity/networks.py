from itertools import product
import numpy as np

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
        connectivity = matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]].mean()
        yield network_a, network_b, connectivity

def block_block(matrix, network_to_idx):
    for network_a, network_b in product(network_to_idx.index, network_to_idx.index):
        loc_a, loc_b = network_to_idx[network_a], network_to_idx[network_b]
        block = matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]]
        n_positive_edges = np.count_nonzero(block > 0)
        n_negative_edges = np.count_nonzero(block < 0)
            
        block_size = (loc_a[1] - loc_a[0]) * (loc_b[1] - loc_b[0])
        #block_activation = (n_negative_edges + n_negative_edges) / block_size
        block_activation = (n_negative_edges > 0) or (n_positive_edges > 0)

        yield network_a, network_b, n_positive_edges, n_negative_edges, block_activation