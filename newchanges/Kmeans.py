import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv('dataset_8.csv')

# Concatenate plaintext and encrypted text for clustering
texts = data['Plaintext'] + ' ' + data['Encryptedtext']

# Feature extraction using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

best_score = -1
best_k = 0
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_k = k

kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X)

cluster_labels = kmeans.predict(X)

# Assign a mode to each cluster based on the most frequent encryption mode in that cluster
cluster_modes = {}
for cluster_label, mode in zip(cluster_labels, data['Mode']):
    if cluster_label not in cluster_modes:
        cluster_modes[cluster_label] = {}
    if mode not in cluster_modes[cluster_label]:
        cluster_modes[cluster_label][mode] = 0
    cluster_modes[cluster_label][mode] += 1

# Determine the mode for each cluster
cluster_assigned_modes = {}
for cluster_label, modes in cluster_modes.items():
    cluster_assigned_modes[cluster_label] = max(modes, key=modes.get)

print("\nCluster Assigned Modes:")
for cluster_label, mode in cluster_assigned_modes.items():
    print(f"Cluster {cluster_label}: {mode}")
