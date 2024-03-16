import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('dataset_8.csv')

# Prepare the data
X = data[['Plaintext', 'Encryptedtext']]
y = data['Mode']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Plaintext'] + X_train['Encryptedtext'])
X_test_vectorized = vectorizer.transform(X_test['Plaintext'] + X_test['Encryptedtext'])

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

# Hyperparameter tuning using grid search
param_grid = {
    'RandomForest': {'n_estimators': [50, 100, 200]},
    'GradientBoosting': {'n_estimators': [50, 100, 200]},
    'LogisticRegression': {'C': [0.1, 1, 10]}
}

for name, classifier in classifiers.items():
    grid_search = GridSearchCV(classifier, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_vectorized, y_train)
    best_classifier = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    print(f"Best {name} classifier: {best_classifier}")
    print(f"Best {name} classifier accuracy: {best_accuracy}")

    # Fit the best classifier on the entire training data
    best_classifier.fit(X_train_vectorized, y_train)

    # Evaluate the classifier on the test set
    accuracy = best_classifier.score(X_test_vectorized, y_test)
    print(f"{name} classifier accuracy on test set: {accuracy}")
