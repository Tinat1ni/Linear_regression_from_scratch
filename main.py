import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None  # placeholder for slope (W in y = W*X + b)
        self.intercept_ = None # placeholder for intercept (b in y = W*X + b)


    def fit(self, X, y):
        # first we need to convert input lists
        # or other iterables to numpy arrays:
        X = np.array(X)
        y = np.array(y)

        # we need to add bias term:
        X_b = np.c_[np.ones(len(X)), X] # we are adding a column
                                        # of ones to x for intercept.
                                        # X_b shape becomes (n_samples, 2)
                                        # the first column (which is all ones)
                                        # corresponds to the intercept term
        '''
        np.c_ -> concatenates arrays column-wise(axis=1)
        it combines multiple 1D arrays into a 2D array by placing them as columns.
        in our case:
           np.ones(len(X)) -> creates a column of ones (for the intercept term)
           X is the original input feature column
           
        The result is a 2D array with shape (n_samples, 2), WHERE:
          column 0 is all ones (bias term of intercept)
          column 1 is the original feature values
        
        This augmented matrix X_b is needed to apply the normal equation for linear regression
        '''

        # now we will apply the Normal Equation to compute the best-fitting parameters.
        # formula : theta = (X_b.T * X_b)^(-1) * X_b.T * y
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        '''
        X_b.T -> transpose of X_b changes shape from (n_samples, 2) to (2, n_samples)
        X_b.T.dot(X_b) -> it will give us dot product of X_b transpose and X_b (X_b.T @ X_b), it will be 2x2 matrix
        np.linalg.inv() -> this computes the inverse of that matrix (The inverse is a matrix that, when multiplied by
                                                                      the original matrix, gives the identity matrix)
        for example: A @ A_inv = I, where I is the identity matrix with 1s on the diagonal and 0s elsewhere.
        
        .dot(X_b.T) -> this multiplies the inverse with X_b.T and gives us a matrix of shape (2, n_samples)
        .dot(y) -> finally multiplies with y (a vector of shape (n_samples,)), resulting in theta (shape: (2,))
        
        The result theta_best is a numpy array : [intercept, slope]
        '''

        self.intercept_ = theta_best[0]   # corresponds to the intercept (b in y = W*X + b)
        self.coef_ = theta_best[1]  # corresponds to the slope (W in y = W*X + b)


    def predict(self,X):
        X = np.array(X)
        return self.intercept_ + self.coef_*X


X = [1,2,3,4,5,6,7,8,9,10]
y = [10,20,30,40,50,60,70,80,90,100]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([20,30,2])

print(f'b: {model.intercept_}')
print(f'W: {model.coef_}')
print(f'prediction: {prediction}')