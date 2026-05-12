import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Linear Regression
data = fetch_california_housing(as_frame=True)

X = data.data[['AveRooms']]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression MSE:",
      mean_squared_error(y_test, y_pred))

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted")

plt.title("Linear Regression")
plt.xlabel("AveRooms")
plt.ylabel("Price")
plt.legend()
plt.show()


# Polynomial Regression
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

cols = ["MPG","Cylinders","Displacement","Horsepower",
        "Weight","Acceleration","Model Year","Origin","Car Name"]
df = pd.read_csv(url, sep=r'\s+',
                 names=cols, na_values="?")

df.dropna(inplace=True)

X = df[['Horsepower']].astype(float)
y = df['MPG']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

print("Polynomial Regression MSE:",
      mean_squared_error(y, y_pred))

plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, label="Polynomial Fit")

plt.title("Polynomial Regression")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()