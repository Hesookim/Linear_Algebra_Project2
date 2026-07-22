import numpy as np

def least_squares(A, b):
    """
    Solve the linear least squares problem using the Moore‑Penrose pseudo‑inverse.

    Given an overdetermined system Ax ≈ b (more equations than unknowns),
    this function finds the solution x that minimizes the Euclidean norm ||Ax - b||₂.
    It uses the pseudo‑inverse A⁺, computed via singular value decomposition (SVD)
    internally by numpy.linalg.pinv, which is a robust and general approach.

    Parameters:

    A : numpy.ndarray
        Feature matrix of shape (m, n), where m = number of samples, n = number of features.
    b : numpy.ndarray
        Target vector of shape (m,).

    Returns:
    
    coefficients : numpy.ndarray
        Model coefficients including the intercept as the first element.
        Shape (n+1,).
    predictions : numpy.ndarray
        Predicted values for the input data, shape (m,).
    """

    # Add a column of ones to the feature matrix to account for the intercept term
    # This transforms A from (m, n) to (m, n+1), where the first column is all 1's.
    # The coefficient corresponding to this column will be the bias/intercept.
    A_bias = np.c_[np.ones(A.shape[0]), A]

    # Compute the Moore‑Penrose pseudo‑inverse of the augmented matrix.
    # np.linalg.pinv uses SVD and automatically handles rank deficiency.
    # The pseudo‑inverse satisfies: A_bias * A_pinv * A_bias = A_bias,
    # and A_pinv * A_bias * A_pinv = A_pinv.
    # The least‑squares solution is then given by x = A_pinv @ b.
    A_pinv = np.linalg.pinv(A_bias)

    # Compute the model coefficients (including intercept) by multiplying
    # the pseudo‑inverse with the target vector.
    coefficients = A_pinv @ b

    # Compute the predicted values using the augmented feature matrix and
    # the obtained coefficients. This is essentially the fitted values.
    predictions = A_bias @ coefficients

    return coefficients, predictions