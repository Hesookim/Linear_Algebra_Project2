import numpy as np

def qr_solver(A, b):
    """
    Solve the linear least squares problem using QR decomposition.

    Given an overdetermined system Ax ≈ b, this function finds the solution x
    that minimizes ||Ax - b||₂ by decomposing A into Q and R,
    then solving R x = Qᵀ b.

    This method is numerically stable and efficient for least squares problems.

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
    # This transforms A from (m, n) to (m, n+1)
    A_bias = np.c_[np.ones(A.shape[0]), A]

    # Get dimensions of the augmented matrix
    m, n = A_bias.shape   # n = number of features + 1 (includes bias)

    # Compute the QR decomposition of the augmented matrix
    # A_bias = Q * R, where:
    # - Q is an orthogonal matrix (m x m) in full mode, or (m, n) in reduced mode
    # - R is an upper triangular matrix (m x n) in full mode, or (n x n) in reduced mode
    # We use 'reduced' mode to get an (n x n) R matrix for direct solving
    Q, R = np.linalg.qr(A_bias, mode='reduced')

    # In reduced mode, R is (n x n) and Q is (m x n)
    # The least squares solution solves: R x = Qᵀ b
    # Since R is upper triangular, we can solve directly using back-substitution
    coefficients = np.linalg.solve(R, Q.T @ b)

    # Compute predictions using the augmented matrix and coefficients
    predictions = A_bias @ coefficients

    return coefficients, predictions