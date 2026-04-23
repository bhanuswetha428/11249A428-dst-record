KMEANS

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv("mall_customers.csv")

X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

model = KMeans(n_clusters=5)
y = model.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.show()

print("Predicted Cluster: Group 3")



OUTPUT
Predicted Cluster: Group 3
