import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Decision Tree Plot
plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)

plt.show()

# New sample prediction
sample = [X_test[0]]

prediction = model.predict(sample)

print("Predicted Class:",
      data.target_names[prediction[0]])
