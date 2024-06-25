import random
import string
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import csv

def generate_sentence():
    characters = string.ascii_letters + " " + string.digits + string.punctuation
    sentence = ''
    while len(sentence) < 50:
        char = random.choice(characters)
        sentence += char
    return sentence

# Function to encrypt plaintext using ECB mode
def encrypt_ecb(key, plaintext):
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded_plaintext) + encryptor.finalize()

# Function to encrypt plaintext using CBC mode
def encrypt_cbc(key, plaintext,iv):
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(padded_plaintext) + encryptor.finalize()

# Function to encrypt plaintext using OFB mode
def encrypt_ofb(key, plaintext,iv):
    cipher = Cipher(algorithms.AES(key), modes.OFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(plaintext)

# Single key for encryption
key = os.urandom(16)

num_sentences = 10000

data = []

for i in range(num_sentences):
    plain_text = generate_sentence()
    iv = os.urandom(16)
    for j in range(3):
        if j == 0:
            Encryptedtext = encrypt_ecb(key, plain_text.encode())
            data.append({'Plaintext':plain_text,'key':key,'iv':'NONE','Encryptedtext':Encryptedtext, "Mode":"ECB"})
        elif j == 1:
            Encryptedtext = encrypt_cbc(key, plain_text.encode(),iv)
            data.append({'Plaintext':plain_text,'key':key,'iv':iv,'Encryptedtext':Encryptedtext, "Mode":"CBC"})
    
        else:
            Encryptedtext = encrypt_ofb(key, plain_text.encode(),iv)
            data.append({'Plaintext':plain_text,'key':key,'iv':iv,'Encryptedtext':Encryptedtext, "Mode":"OFB"})

field_names=['Plaintext','key','iv','Encryptedtext',"Mode"]

# Save encrypted data into a CSV file
with open('Largedatatest.csv', mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    
    # Write header
    writer.writeheader()
    
    # Write rows
    for row in data:
        writer.writerow(row)

print('dataset created')
