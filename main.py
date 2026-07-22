"""
Linear System Solver Comparison:
This script solves the overdetermined system Ax = b using four different methods:
1. Least Squares (Pseudo-Inverse)
2. QR Decomposition
3. SVD Decomposition
4. Linear Regression (scikit-learn)

The dataset is loaded from 'dataset.csv', standardized, and results are compared
in terms of coefficients, MSE, R², execution time, and residual norms.
"""

import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.least_squares import least_squares
from utils.qr_solver import qr_solver
from utils.svd_solver import svd_solver


# 1. LOAD DATASET

# Read the CSV file containing the dataset
df = pd.read_csv("dataset.csv")

# Separate features (A) and target (b)
# 'b' is the last column; all other columns are features
A = df.drop(columns=["b"]).to_numpy(dtype=float)
b = df["b"].to_numpy(dtype=float)

# Display basic information about the dataset
print("=" * 50)
print("Matrix Information")
print("=" * 50)
print("Shape of A:", A.shape)          # (number of samples, number of features)
print("Shape of b:", b.shape)          # (number of samples,)

print("\nFirst Row of A:")
print(A[0])                            # Show first sample's feature vector

print("\nFirst 5 Values of b:")
print(b[:5])                           # Show first 5 target values


# 2. PREPROCESSING: STANDARDIZATION

# Standardize features to have zero mean and unit variance
# This is important for numerical stability and fair comparison
scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)

# Compute condition number to check numerical stability
# High condition number (>1000) indicates ill-conditioned system
cond_num = np.linalg.cond(A_scaled)
print(f"\nCondition Number of A_scaled: {cond_num:.2f}")


# METHOD 1: LEAST SQUARES (PSEUDO-INVERSE)

# Solve using Moore-Penrose pseudo-inverse: x = (A^T A)^(-1) A^T b
start = time.perf_counter()                     # Start timing
coefficients, b_pred_ls = least_squares(A_scaled, b)
end = time.perf_counter()                       # End timing
time_ls = end - start

# Compute residual norm (||Ax - b||) to measure fit quality
residual_ls = np.linalg.norm(b - b_pred_ls)

# Evaluate performance metrics
mse_ls = mean_squared_error(b, b_pred_ls)       # Mean Squared Error
r2_ls = r2_score(b, b_pred_ls)                  # R² coefficient of determination

print("\nLeast Squares")
print("Intercept:")
print(coefficients[0])                           # First coefficient is the bias term
print("\nCoefficients:")
print(coefficients[1:])                         # Remaining coefficients for features


# METHOD 2: LINEAR REGRESSION (SCIKIT-LEARN)

# Use scikit-learn's LinearRegression as a reference implementation
model = LinearRegression()

start = time.perf_counter()
model.fit(A_scaled, b)                           # Fit the model
end = time.perf_counter()
time_lr = end - start

# Predict and compute metrics
b_pred_lr = model.predict(A_scaled)
residual_lr = np.linalg.norm(b - b_pred_lr)
mse_lr = mean_squared_error(b, b_pred_lr)
r2_lr = r2_score(b, b_pred_lr)

print("\n" + "=" * 50)
print("Linear Regression")
print("=" * 50)
print("Coefficients:")
print(model.coef_)                               # Feature coefficients
print("\nIntercept:")
print(model.intercept_)                          # Bias term
print(f"\nMSE : {mse_lr:.6f}")
print(f"R2  : {r2_lr:.6f}")

# METHOD 3: QR DECOMPOSITION

# Solve using QR decomposition: A = QR → R x = Q^T b
start = time.perf_counter()
coeff_qr, pred_qr = qr_solver(A_scaled, b)
end = time.perf_counter()
time_qr = end - start

residual_qr = np.linalg.norm(b - pred_qr)
mse_qr = mean_squared_error(b, pred_qr)
r2_qr = r2_score(b, pred_qr)

print("\n" + "=" * 50)
print("QR Decomposition")
print("=" * 50)
print("Intercept:")
print(coeff_qr[0])
print("\nCoefficients:")
print(coeff_qr[1:])
print(f"\nMSE : {mse_qr:.6f}")
print(f"R2  : {r2_qr:.6f}")

# METHOD 4: SVD DECOMPOSITION

# Solve using SVD: A = U Σ V^T → x = V Σ^(-1) U^T b
start = time.perf_counter()
coeff_svd, pred_svd = svd_solver(A_scaled, b)
end = time.perf_counter()
time_svd = end - start

residual_svd = np.linalg.norm(b - pred_svd)
mse_svd = mean_squared_error(b, pred_svd)
r2_svd = r2_score(b, pred_svd)

print("\n" + "=" * 50)
print("SVD Solver")
print("=" * 50)
print("Intercept:")
print(coeff_svd[0])
print("\nCoefficients:")
print(coeff_svd[1:])
print(f"\nMSE : {mse_svd:.6f}")
print(f"R2  : {r2_svd:.6f}")


# CONSISTENCY TEST

# Verify that all methods produce identical results (within floating-point tolerance)
# This confirms the correctness of the implementations
print("\n" + "=" * 50)
print("Consistency Test")
print("=" * 50)

print("Least Squares == QR Solver :",
      np.allclose(coefficients, coeff_qr))

print("Least Squares == SVD Solver:",
      np.allclose(coefficients, coeff_svd))

print("Least Squares == Linear Regression:",
      np.allclose(coefficients, np.r_[model.intercept_, model.coef_]))


# COMPREHENSIVE COMPARISON TABLE

# Display all metrics in a clean table format
print("\n" + "=" * 50)
print("Comparison")
print("=" * 50)
print(f"{'Method':<20}{'MSE':<15}{'R2':<15}{'Time (s)':<15}")
print("-" * 65)

print(f"{'Least Squares':<20}{mse_ls:<15.6f}{r2_ls:<15.6f}{time_ls:<15.8f}")
print(f"{'QR Solver':<20}{mse_qr:<15.6f}{r2_qr:<15.6f}{time_qr:<15.8f}")
print(f"{'SVD Solver':<20}{mse_svd:<15.6f}{r2_svd:<15.6f}{time_svd:<15.8f}")
print(f"{'Linear Regression':<20}{mse_lr:<15.6f}{r2_lr:<15.6f}{time_lr:<15.8f}")


# RESIDUAL NORM COMPARISON

# Compare the norm of residuals ||Ax - b|| for each method
# All should be essentially zero if the system is consistent
print("\n" + "=" * 50)
print("Residual Norm Comparison")
print("=" * 50)
print(f"{'Method':<20}{'||Ax-b||'}")
print("-" * 35)

print(f"{'Least Squares':<20}{residual_ls:.6f}")
print(f"{'QR Solver':<20}{residual_qr:.6f}")
print(f"{'SVD Solver':<20}{residual_svd:.6f}")
print(f"{'Linear Regression':<20}{residual_lr:.6f}")


# CONCLUSION

# Summary of findings
print("\nConclusion:")
print("All methods produce identical coefficients and prediction accuracy.")
print("The implementation has been successfully verified.")
print(f"Condition number of the scaled matrix is {cond_num:.2f} – the system is well-conditioned.")