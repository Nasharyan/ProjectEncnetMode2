import random
import string
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

def generate_sentence():
    characters = string.ascii_letters + " " + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(50))

def encrypt_data(key, plaintext, mode='ECB', iv=None):
    padder = padding.PKCS7(256).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
    if mode == 'ECB':
        cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    elif mode == 'CBC':
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    elif mode == 'OFB':
        cipher = Cipher(algorithms.AES(key), modes.OFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_plaintext) + encryptor.finalize()
    return iv + encrypted_data if mode != 'ECB' else encrypted_data