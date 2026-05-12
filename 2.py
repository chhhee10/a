import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
california_data = fetch_california_housing(as_frame=True)
data = california_data.frame

# Correlation matrix
correlation_matrix = data.corr()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Pair plot
sns.pairplot(data)
plt.show()