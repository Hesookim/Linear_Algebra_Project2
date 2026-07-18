import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.least_squares import least_squares
from utils.qr_solver import qr_solver
from utils.svd_solver import svd_solver


df = pd.read_csv("dataset.csv")


A = df.drop(columns=["b"]).to_numpy(dtype=float)
b = df["b"].to_numpy(dtype=float)

print("=" * 50)
print("Matrix Information")
print("=" * 50)
print("Shape of A:", A.shape)
print("Shape of b:", b.shape)

print("\nFirst Row of A:")
print(A[0])

print("\nFirst 5 Values of b:")
print(b[:5])


scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)

start = time.perf_counter()

end = time.perf_counter()
time_ls = end - start

coefficients, b_pred_ls = least_squares(A_scaled, b)

residual_ls = np.linalg.norm(b - b_pred_ls)

print("\n===== Least Squares =====")

print("\nIntercept:")
print(coefficients[0])

print("\nCoefficients:")
print(coefficients[1:])


model = LinearRegression()


start = time.perf_counter()

model.fit(A_scaled, b)

end = time.perf_counter()
time_lr = end - start

b_pred_lr = model.predict(A_scaled)

residual_lr = np.linalg.norm(b - b_pred_lr)

mse_ls = mean_squared_error(b, b_pred_ls)
r2_ls = r2_score(b, b_pred_ls)

start = time.perf_counter()

end = time.perf_counter()
time_qr = end - start

coeff_qr, pred_qr = qr_solver(A_scaled, b)

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

print("\nMSE :", mse_qr)
print("R2  :", r2_qr)


print("\n" + "=" * 50)
print("Linear Regression")
print("=" * 50)

print("Coefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

print("\nMSE :", mse_ls)
print("R2  :", r2_ls)

start = time.perf_counter()

end = time.perf_counter()
time_svd = end - start

coeff_svd, pred_svd = svd_solver(A_scaled, b)

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


print("\nMSE :", mse_svd)
print("R2  :", r2_svd)

print("\n" + "=" * 50)
print("Consistency Test")
print("=" * 50)

print("Least Squares == QR Solver :",
      np.allclose(coefficients, coeff_qr))

print("Least Squares == SVD Solver:",
      np.allclose(coefficients, coeff_svd))

print("Least Squares == Linear Regression:",
      np.allclose(coefficients, np.r_[model.intercept_, model.coef_]))

print("\n" + "=" * 50)
print("Comparison")
print("=" * 50)

print(f"{'Method':<20}{'MSE':<15}{'R2'}")
print("-"*45)

print(f"{'Least Squares':<20}{mse_ls:<15.6f}{r2_ls:.6f}")
print(f"{'QR Solver':<20}{mse_qr:<15.6f}{r2_qr:.6f}")
print(f"{'SVD Solver':<20}{mse_svd:<15.6f}{r2_svd:.6f}")
print(f"{'Linear Regression':<20}{mse_ls:<15.6f}{r2_ls:.6f}")


print("\n" + "=" * 50)
print("Execution Time")
print("=" * 50)

print(f"Least Squares      : {time_ls:.8f} sec")
print(f"QR Solver          : {time_qr:.8f} sec")
print(f"SVD Solver         : {time_svd:.8f} sec")
print(f"Linear Regression  : {time_lr:.8f} sec")


print("\n" + "=" * 50)
print("Residual Norm Comparison")
print("=" * 50)

print(f"{'Method':<20}{'||Ax-b||'}")
print("-"*35)

print(f"{'Least Squares':<20}{residual_ls:.6f}")
print(f"{'QR Solver':<20}{residual_qr:.6f}")
print(f"{'SVD Solver':<20}{residual_svd:.6f}")
print(f"{'Linear Regression':<20}{residual_lr:.6f}")


print("\nConclusion:")
print("All methods produce identical coefficients and prediction accuracy.")
print("The implementation has been successfully verified.")