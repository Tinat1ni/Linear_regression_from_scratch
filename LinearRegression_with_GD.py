import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate # step size for each update in gradient descent
        self.n_iterations = n_iterations   # number of iterations (epochs) for gradient descent

        self.coef_ = None  # will hold the final w
        self.intercept_ = None # will hold the final b
        self.losses = []  # to store the loss at each iteration


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)

        w = 0.0
        b = 0.0
        '''
        we start with a line y = 0*X + 0  (a flat line  at y=0)
        the goal is to gradually update w and b to minimize the loss.
        '''
        for i in range(self.n_iterations):
            # make predictions using current w and b
            y_pred = w*X + b
            # calculate prediction error
            error = y - y_pred

            loss = np.mean(error**2)
            self.losses.append(loss) # store loss in a list so we can visualize it later

            '''
            compute the gradients (partial derivatives) of the loss function
            with respect to w and b. these tell us the slope of the loss function
            and help decide how to update our parameters to reduce the error. 
            '''
            dw = -2*np.dot(error,X) / n # dw tells us how sensitive is the loss to changes in w
            db = -2*np.sum(error) / n   # db tells us how sensitive is the loss to changes in b

            # gradient descent step: update w and b in the direction that reduces loss
            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

        self.coef_ = w
        self.intercept_ = b



    def predict(self, X):
        X = np.array(X)
        return self.coef_ * X + self.intercept_

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('iteration')
        plt.ylabel('loss(MSE)')
        plt.grid(True)
        plt.show()


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

prediction = model.predict([20, 30, 2])
print(f'prediction:{prediction}')
print(f'final slope(w): {model.coef_}')
print(f'final intercept(b): {model.intercept_}')

model.plot_loss()