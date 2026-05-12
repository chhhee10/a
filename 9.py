import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
print("Loading Olivetti Face Dataset...")

data = fetch_olivetti_faces()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Naive Bayes Accuracy:",
      accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Display few predictions
fig, axes = plt.subplots(2,5, figsize=(10,5))

for i, ax in enumerate(axes.flat):

    ax.imshow(X_test[i].reshape(64,64), cmap="gray")

    ax.set_title(f"P:{y_pred[i]} T:{y_test[i]}")

    ax.axis("off")

plt.show()

# New sample prediction
sample = [X_test[0]]

prediction = model.predict(sample)

print("\nPredicted Class:", prediction[0])
print("True Class:", y_test[0])