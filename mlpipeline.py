import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, SVC

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data[['Plaintext', 'Encryptedtext']]
    y = data['Mode']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train['Plaintext'] + X_train['Encryptedtext'])
    X_test_vectorized = vectorizer.transform(X_test['Plaintext'] + X_test['Encryptedtext'])
    return X_train_vectorized, X_test_vectorized

def train_and_evaluate_model(X_train, y_train, X_test, y_test, classifier, params):
    grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average=None),
        'Recall': recall_score(y_test, y_pred, average=None),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    }
    return metrics, grid_search.best_params_