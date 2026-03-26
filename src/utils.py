import os
import numpy as np


def generate_fixed_subscription_matrix(n, k, sub_prob, seed):
    """
    Generate a binary subscription matrix.
    matrix[i, j] = 1 means subscriber j subscribes to topic i
    """
    rng = np.random.default_rng(seed)
    matrix = (rng.random((n, k)) < sub_prob).astype(int)

    # Ensure each topic has at least one subscriber
    for i in range(n):
        if matrix[i].sum() == 0:
            j = rng.integers(0, k)
            matrix[i, j] = 1

    return matrix


def save_matrix(matrix, filepath):
    """
    Save matrix to file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, matrix)