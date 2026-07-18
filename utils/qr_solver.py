import numpy as np


def qr_solver(A, b):
    """
    Solve Least Squares using QR Decomposition.

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

    # QR decomposition
    Q, R = np.linalg.qr(A_bias)

    # Solve R x = Q^T b
    coefficients = np.linalg.solve(R, Q.T @ b)

    # Prediction
    predictions = A_bias @ coefficients

    return coefficients, predictions