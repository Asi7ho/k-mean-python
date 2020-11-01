
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Dataset
dataSet = pd.read_csv('/Users/yanndebain/MEGA/MEGAsync/Code/Data Science/ML/Clustering/Mall_Customers.csv')

X = dataSet.iloc[:, 3:].values

# Elbow method
WCSS = []
K = np.linspace(1, 10, 10, dtype=int)
for i in range(len(K)):
    kmeans = KMeans(K[i], init='k-means++', random_state=0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

plt.plot(K, WCSS)
plt.title('Choice of k-cluster (Elbow\'s method)')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()

# Model -> Choice of K = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_cluster = kmeans.fit_predict(X)

# Visualization
plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], c = 'red', label = 'cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], c = 'green', label = 'cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], c = 'blue', label = 'cluster 3')
plt.scatter(X[y_cluster == 3, 0], X[y_cluster == 3, 1], c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_cluster == 4, 0], X[y_cluster == 4, 1], c = 'magenta', label = 'cluster 5')
plt.legend()
plt.xlabel('Salary')
plt.ylabel('Spending Score')
plt.title('Client Clusters')
plt.show()


