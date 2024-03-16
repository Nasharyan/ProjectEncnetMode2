import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('dataset_8.csv')

X = data[['Plaintext','Encryptedtext']]
y = data['Mode'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Plaintext'] + X_train['Encryptedtext'])
X_test_vectorized = vectorizer.transform(X_test['Plaintext'] + X_test['Encryptedtext'])

classifiers = {}

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vectorized, y_train)
classifiers['RandomForest'] = rf_classifier

gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_vectorized, y_train)
classifiers['GradientBoosting'] = gb_classifier

lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_vectorized, y_train)
classifiers['LogisticRegression'] = lr_classifier

for model_name, classifier in classifiers.items():
    # Predictions on the testing set
    y_pred = classifier.predict(X_test_vectorized)
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy:", accuracy)

def decrypt_ecb(ciphertext, key):
    # Create an AES ECB cipher object
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the decrypted message
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_message = unpadder.update(decrypted_message)
    unpadded_message += unpadder.finalize()

    return unpadded_message

def decrypt_cbc(ciphertext, key, iv):
    # Create an AES CBC cipher object
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the decrypted message
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_message = unpadder.update(decrypted_message)
    unpadded_message += unpadder.finalize()

    return unpadded_message

def decrypt_ofb(ciphertext, key, iv):
    # Create an AES OFB cipher object
    cipher = Cipher(algorithms.AES(key), modes.OFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()

    return decrypted_message
