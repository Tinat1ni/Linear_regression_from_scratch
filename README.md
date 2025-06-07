# Linear Regression from Scratch with NumPy

This project demonstrates how to implement **Linear Regression** from scratch in Python using only NumPy.

## How it Works

The `LinearRegression` class does the following:

1. **Adds a bias term (intercept)** to input data
2. **Applies the Normal Equation**:
   \[
   \theta = (X^T X)^{-1} X^T y
   \]
3. Stores `intercept_` and `coef_`
4. Can predict output for new values
