import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values
np.random.seed(42)
x = np.random.rand(100).reshape(-1,1)

# Labels for first 50 points
y = np.array([1 if i <= 0.5 else 2 for i in x[:50].flatten()])

# Training and testing data
X_train = x[:50]
X_test = x[50:]
y_train = y

# Print training data
print("Training Data:")
for i in range(50):
    print(f"x[{i+1}] = {X_train[i][0]:.4f}, Class: {y_train[i]}")

# K values
k_values = [1,2,3,4,5,20,30]

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(f"\nk = {k} Predictions:")
    print(y_pred)

    # Plot
    plt.scatter(X_train, y_train, label="Train Data")
    plt.scatter(X_test, y_pred, label="Predicted Data")

    plt.axhline(y=1.5, linestyle='--')
    plt.title(f"KNN with k={k}")
    plt.xlabel("x values")
    plt.ylabel("Class")
    plt.legend()
    plt.show()