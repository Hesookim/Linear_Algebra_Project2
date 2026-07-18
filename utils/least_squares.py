import numpy as np


def least_squares(A, b):
    """
    Solve Ax = b using Least Squares (Pseudo-Inverse)

    Parameters
    ----------
    A : numpy.ndarray
        Feature matrix
    b : numpy.ndarray
        Target vector

    Returns
    -------
    coefficients : numpy.ndarray
        Model coefficients
    predictions : numpy.ndarray
        Predicted values
    """

    # Add bias column
    A_bias = np.c_[np.ones(A.shape[0]), A]

    # Compute pseudo-inverse
    A_pinv = np.linalg.pinv(A_bias)

    # Compute coefficients
    coefficients = A_pinv @ b

    # Predictions
    predictions = A_bias @ coefficients

    return coefficients, predictions