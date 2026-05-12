import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading Dataset...")

data = load_breast_cancer()

X = data.data
y = data.target

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
print("Performing K-Means Clustering...")

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Predictions
labels = kmeans.labels_

# Accuracy
print("Accuracy:",
      accuracy_score(y, labels))

# PCA for visualization
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8,6))

plt.scatter(X_pca[:,0],
            X_pca[:,1],
            c=labels)

plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()