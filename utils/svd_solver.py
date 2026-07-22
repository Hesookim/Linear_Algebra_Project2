import numpy as np

def svd_solver(A, b):
    """
    Solve the linear least squares problem using Singular Value Decomposition (SVD).

    Given an overdetermined system Ax ≈ b, this function finds the solution x
    that minimizes ||Ax - b||₂ using the Moore-Penrose pseudo-inverse
    computed via SVD: x = V Σ⁺ Uᵀ b, where Σ⁺ is the pseudo-inverse of Σ.

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

    # Compute the Singular Value Decomposition of the augmented matrix
    # A_bias = U * diag(S) * Vᵀ, with S sorted in descending order
    # full_matrices=False gives the "economy" decomposition for efficiency
    U, S, VT = np.linalg.svd(A_bias, full_matrices=False)

    # Compute the pseudo-inverse of S: invert non-zero singular values
    # For numerical stability, we could filter out very small values,
    # but here we assume S has no zero or extremely small entries.
    # If needed, one could add: S_inv = np.diag(1 / S[S > tol])
    S_inv = np.diag(1 / S)   # S is 1D, diag makes a square diagonal matrix

    # The pseudo-inverse of A_bias is: V * S_inv * Uᵀ
    # VT is Vᵀ, so V = VT.T, and U is U.
    A_pinv = VT.T @ S_inv @ U.T

    # Solve for coefficients: x = A⁺ b
    coefficients = A_pinv @ b

    # Compute predictions using the augmented matrix and coefficients
    predictions = A_bias @ coefficients

    return coefficients, predictions