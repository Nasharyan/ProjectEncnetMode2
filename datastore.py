import csv

def save_data_to_csv(data, filename):
    field_names = ['Plaintext', 'key', 'iv', 'Encryptedtext', 'Mode']
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)