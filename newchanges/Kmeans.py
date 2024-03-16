import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Separate features (encrypted texts) and labels (encryption modes)
X_classifier = data[['Plaintext', 'Encryptedtext']]
y_classifier = data['Mode']

X_train_classifier, X_test_classifier, y_train_classifier, y_test_classifier = train_test_split(X_classifier, y_classifier, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorizer
vectorizer_classifier = TfidfVectorizer()
X_train_vectorized = vectorizer_classifier.fit_transform(X_train_classifier['Plaintext'] + X_train_classifier['Encryptedtext'])
X_test_vectorized = vectorizer_classifier.transform(X_test_classifier['Plaintext'] + X_test_classifier['Encryptedtext'])

# Model training using Random Forest Classifier for each encryption mode
classifiers = {}
for mode in y_train_classifier.unique():
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_vectorized, (y_train_classifier == mode).astype(int))
    classifiers[mode] = classifier

# Dictionary to store accuracies
accuracies = {}

# Loop through each classifier and calculate accuracy
for mode, classifier in classifiers.items():
    # Predictions on the testing set
    y_pred = classifier.predict(X_test_vectorized)
    
    # Evaluate model performance
    accuracy = accuracy_score((y_test_classifier == mode).astype(int), y_pred)
    accuracies[mode] = accuracy

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel('Encryption Mode')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Encryption Modes')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()
