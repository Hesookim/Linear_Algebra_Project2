import numpy as np


def svd_solver(A, b):
    """
    Solve Least Squares using Singular Value Decomposition (SVD)

    Parameters
    ----------
    A : ndarray
        Feature matrix

    b : ndarray
        Target vector

    Returns
    -------
    coefficients
    predictions
    """

    # Add bias column
    A_bias = np.c_[np.ones(A.shape[0]), A]

    # SVD decomposition
    U, S, VT = np.linalg.svd(A_bias, full_matrices=False)

    # Compute pseudo-inverse using SVD
    S_inv = np.diag(1 / S)

    A_pinv = VT.T @ S_inv @ U.T

    # Coefficients
    coefficients = A_pinv @ b

    # Predictions
    predictions = A_bias @ coefficients

    return coefficients, predictions