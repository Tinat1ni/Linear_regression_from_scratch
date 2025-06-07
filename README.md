# Linear Regression from Scratch with NumPy

This project demonstrates how to implement **Linear Regression** from scratch in Python using only NumPy.  
It includes two methods:

- `LinearRegression` using the **Normal Equation** (found in `main.py`)
- `LinearRegression` using **Gradient Descent** (found in `LinearRegression_with_GD.py`)

---

##  How the Normal Equation Method Works (`main.py`)

This approach solves for the optimal parameters analytically using the **Normal Equation**:

\[
\theta = (X^T X)^{-1} X^T y
\]

### Steps:
1. Adds a **bias term (intercept)** to input data
2. Computes parameters using the closed-form Normal Equation
3. Stores `intercept_` and `coef_`
4. Can predict outputs for new input values


### 2. Gradient Descent Method (`LinearRegression_with_GD.py`)

- Iteratively updates parameters to minimize the Mean Squared Error (MSE).
- Requires setting a learning rate and number of iterations.
- Can be used on large datasets where Normal Equation is computationally expensive.
- Provides insight into how machine learning models learn over time through the loss curve.

**How it works:**

- Starts with initial weights (w=0, b=0).
- Calculates prediction error and gradients (how loss changes with respect to weights).
- Updates weights in small steps opposite to the gradient to reduce error.
- Stores loss values for each iteration to visualize convergence.
- After training, provides slope and intercept for predictions.

---


