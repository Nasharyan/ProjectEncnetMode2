# Import the necessary modules created earlier
import os

# Data Generation and Encryption
key = os.urandom(16)  # Encryption key
num_sentences = 350  # Number of sentences to generate
data = []

for i in range(num_sentences):
    plain_text = generate_sentence()
    iv = os.urandom(16)  # Initialization vector for CBC and OFB
    encrypted_ecb = encrypt_data(key, plain_text.encode(), mode='ECB')
    encrypted_cbc = encrypt_data(key, plain_text.encode(), mode='CBC', iv=iv)
    encrypted_ofb = encrypt_data(key, plain_text.encode(), mode='OFB', iv=iv)
    
    data.append({'Plaintext': plain_text, 'key': key, 'iv': 'NONE', 'Encryptedtext': encrypted_ecb, "Mode": "ECB"})
    data.append({'Plaintext': plain_text, 'key': key, 'iv': iv, 'Encryptedtext': encrypted_cbc, "Mode": "CBC"})
    data.append({'Plaintext': plain_text, 'key': key, 'iv': iv, 'Encryptedtext': encrypted_ofb, "Mode": "OFB"})

# Data Storage
save_data_to_csv(data, 'encrypted_data.csv')

# Machine Learning Pipeline
X, y_encoded = load_and_preprocess_data('encrypted_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)

classifiers = {
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(),
        'params': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'n_estimators': [100, 200]}
    },
    'SVM-RBF': {
        'model': SVC(kernel='rbf'),
        'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(),
        'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1]}
    }
}

# Train, Evaluate and Display Results
for name, specs in classifiers.items():
    metrics, best_params = train_and_evaluate_model(X_train_vectorized, y_train, X_test_vectorized, y_test, specs['model'], specs['params'])
    print(f"Results for {name}:")
    print("Accuracy:", metrics['Accuracy'])
    print("Precision:", metrics['Precision'])
    print("Recall:", metrics['Recall'])
    print("F1 Score:", metrics['F1 Score'])
    print("Best Parameters:", best_params)
    print("-" * 40)