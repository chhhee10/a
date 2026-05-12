import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression Function
def lwr(x, y, query, tau):

    X = np.c_[np.ones(len(x)), x]
    q = np.array([1, query])

    w = np.exp(-((x - query)**2) / (2 * tau**2))
    W = np.diag(w)

    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

    return q @ theta

# Dataset
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,2,4,3.5,5,6,7,6.5,8])

# Prediction points
x_query = np.linspace(1,10,100)
tau = 1.0

# Predictions
y_pred = np.array([lwr(x, y, i, tau) for i in x_query])

# Plot
plt.scatter(x, y, label="Data Points")
plt.plot(x_query, y_pred, label="LWR")

plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()