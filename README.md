# 🧮 Linear System Solver Comparison

> A comprehensive comparison of four numerical methods for solving overdetermined linear systems:  
> **Least Squares (Pseudo-Inverse), QR Decomposition, SVD Decomposition, and Linear Regression (scikit-learn).**

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [How to Run](#how-to-run)
- [Methods Explained](#methods-explained)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Output](#sample-output)
- [Results Interpretation](#results-interpretation)
- [Educational Value](#educational-value)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## 📖 Project Overview

This project implements and compares **four different numerical methods** for solving overdetermined linear systems of the form:

\[
Ax \approx b
\]

where \(A\) is an \(m \times n\) matrix with \(m > n\) (more equations than unknowns). Such systems are common in data fitting, regression analysis, and machine learning. Since an exact solution generally does not exist, we seek the **least squares solution** that minimizes the residual norm:

\[
\min_x \|Ax - b\|_2
\]

The project was developed as part of a **Linear Algebra course** to demonstrate the practical application of matrix decomposition techniques in solving real-world problems.

---

## 🎯 Objectives

- ✅ Load and preprocess a real-world dataset
- ✅ Implement three numerical solvers from scratch:
  - Moore‑Penrose **Pseudo‑Inverse**
  - **QR Decomposition**
  - **SVD Decomposition**
- ✅ Compare results with scikit-learn's `LinearRegression`
- ✅ Evaluate performance using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
  - **Residual Norm** \(\|Ax - b\|\)
  - **Execution Time**
- ✅ Verify consistency across all methods
- ✅ Analyze numerical stability using **Condition Number**

---

## 🧮 Mathematical Background

### The Least Squares Problem

For an overdetermined system \(Ax = b\), we minimize:

\[
\|Ax - b\|_2^2 = \sum_{i=1}^m (a_i^T x - b_i)^2
\]

The solution is given by solving the **normal equations**:

\[
A^T A x = A^T b
\]

However, solving the normal equations directly can be numerically unstable. Instead, we use more robust decomposition methods.

### Three Decomposition Methods

| Method | Mathematical Formulation | Advantages |
| :--- | :--- | :--- |
| **Pseudo‑Inverse** | \(x = A^+ b\), where \(A^+ = V \Sigma^+ U^T\) | Simple, works for rank‑deficient matrices |
| **QR Decomposition** | \(A = QR \Rightarrow R x = Q^T b\) | Numerically stable, efficient |
| **SVD Decomposition** | \(A = U \Sigma V^T \Rightarrow x = V \Sigma^+ U^T b\) | Most robust, handles ill‑conditioned systems |

---

## 📁 Project Structure
