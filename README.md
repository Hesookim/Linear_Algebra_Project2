# Solving Overdetermined Linear Systems Using Linear Algebra

## Project Overview

This project implements multiple numerical methods for solving
**overdetermined systems of linear equations** using concepts from
**Linear Algebra**.

Instead of relying solely on machine learning libraries, the project
develops the mathematical algorithms from scratch, including:

-   Least Squares
-   QR Decomposition
-   Singular Value Decomposition (SVD)

The obtained solutions are compared with **Scikit-Learn's Linear
Regression** implementation to verify correctness and numerical
consistency.

This project was developed as part of a **Linear Algebra** course.

------------------------------------------------------------------------

# Objectives

The main objectives of this project are:

-   Read and preprocess the dataset.
-   Construct matrix **A** and vector **b**.
-   Standardize the input features.
-   Solve the system using Least Squares.
-   Solve the system using QR Decomposition.
-   Solve the system using Singular Value Decomposition (SVD).
-   Compare all methods with Linear Regression.
-   Evaluate prediction accuracy using MSE and R².
-   Compute the residual norm.
-   Measure execution time.
-   Verify numerical consistency between all methods.

------------------------------------------------------------------------

# Project Structure

``` text
Linear_Algebra_Project2
│
├── dataset.csv
├── main.py
├── README.md
│
└── utils
    ├── least_squares.py
    ├── qr_solver.py
    └── svd_solver.py
```

------------------------------------------------------------------------

# Features

## Core Features

-   Dataset loading
-   Matrix construction
-   Feature standardization
-   Least Squares implementation
-   QR Decomposition solver
-   Singular Value Decomposition solver
-   Linear Regression comparison
-   Numerical evaluation

## Advanced Features

### Least Squares Solver

Uses the Moore--Penrose pseudoinverse:

x = A⁺b

to compute the optimal least-squares solution without explicitly
computing the inverse matrix.

### QR Decomposition

Factorizes

A = QR

and solves

Rx = Qᵀb

providing an efficient and numerically stable solution.

### Singular Value Decomposition (SVD)

Factorizes

A = UΣVᵀ

and computes

x = VΣ⁻¹Uᵀb

which is especially robust for ill-conditioned matrices.

### Performance Evaluation

Each method is evaluated using:

-   Mean Squared Error (MSE)
-   Coefficient of Determination (R²)
-   Residual Norm
-   Execution Time

### Consistency Verification

The coefficient vectors produced by all methods are compared using
`numpy.allclose()` to verify numerical equivalence.

------------------------------------------------------------------------

# Mathematical Background

Given

Ax = b

where A ∈ ℝ\^(m×n) and m \> n, the system is generally inconsistent.

The Least Squares problem minimizes

\|\|Ax − b\|\|²

The pseudoinverse solution is

x = A⁺b

Alternative numerical solutions include QR decomposition and Singular
Value Decomposition.

------------------------------------------------------------------------

# Experimental Results

Using the provided dataset, all methods produced identical solutions
(within floating-point precision).

  Method                     MSE         R²   Residual Norm
  ------------------- ---------- ---------- ---------------
  Least Squares         7.358190   0.986123       11.823942
  QR Solver             7.358190   0.986123       11.823942
  SVD Solver            7.358190   0.986123       11.823942
  Linear Regression     7.358190   0.986123       11.823942

------------------------------------------------------------------------

# Execution Time

Measured execution times:

  Method                Time (seconds)
  ------------------- ----------------
  Least Squares             0.00000120
  QR Solver                 0.00000090
  SVD Solver                0.00000210
  Linear Regression         0.00256590

**Insert the execution-time comparison chart here.**

Observation:

-   QR Solver achieved the fastest execution time.
-   All four methods produced identical numerical accuracy.
-   Linear Regression required additional overhead due to the
    Scikit-Learn implementation.

------------------------------------------------------------------------

# Sample Console Output

The console output includes:

-   Matrix Information
-   Least Squares coefficients
-   QR coefficients
-   SVD coefficients
-   Linear Regression coefficients
-   MSE
-   R²
-   Residual Norm
-   Consistency Test
-   Execution Time Comparison

------------------------------------------------------------------------

# Educational Concepts

This project demonstrates practical applications of:

-   Matrix Algebra
-   Least Squares Optimization
-   QR Decomposition
-   Singular Value Decomposition
-   Numerical Linear Algebra
-   Regression Analysis
-   Performance Evaluation

------------------------------------------------------------------------

# Possible Future Improvements

-   Ridge Regression
-   Lasso Regression
-   Cross Validation
-   Polynomial Regression
-   Sparse Matrix Solvers
-   GPU Acceleration
-   Interactive Visualizations

------------------------------------------------------------------------

# Conclusion

The project successfully implemented and compared four numerical
approaches for solving an overdetermined linear system.

All methods produced identical coefficients, prediction accuracy,
residual norms, and nearly identical execution times (except for the
additional overhead of the Scikit-Learn implementation).

The results confirm the mathematical equivalence of Least Squares, QR
Decomposition, SVD, and Linear Regression for this dataset while
highlighting the efficiency of QR decomposition.

------------------------------------------------------------------------

# Author

**Minoo**

Bachelor of Computer Engineering

University of Zanjan

Linear Algebra Course Project

2026
