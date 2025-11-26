import os
import csv
import random
from collections import defaultdict

# Configuration
path = 'recordings/'
train_ratio = 0.90  # => 1 locuteur en test/les autres pour le train

print("=== Séparation Train/Test par locuteur pour AudioMNIST ===\n")

# Dictionnaire pour regrouper les fichiers par locuteur
# Structure: {locuteur: [(file_path, label), ...]}
speaker_files = defaultdict(list)

# On parcourt tous les fichiers audio
print(f"Lecture des fichiers dans {path}...")
file_count = 0
for file in os.listdir(path):
    if file.lower().endswith('.wav'):
        file_path = path + file
        
        # Parser le nom du fichier: chiffre_locuteur_repetition.wav
        try:
            parts = file.replace('.wav', '').split('_')
            if len(parts) >= 3:
                label = parts[0]  # Le chiffre prononcé
                speaker = parts[1]  # Le numéro du locuteur
                
                # Ajouter le fichier à ce locuteur
                speaker_files[speaker].append((file_path, label))
                file_count += 1
            else:
                print(f"Attention: fichier ignoré (format incorrect): {file}")
        except Exception as e:
            print(f"Erreur lors du parsing de {file}: {e}")

print(f"Total de fichiers trouvés: {file_count}")
print(f"Nombre de locuteurs: {len(speaker_files)}\n")

# Séparation des locuteurs en train/test
#########################################
print("\n=== Séparation des locuteurs ===")

# Obtenir la liste des locuteurs
speakers = list(speaker_files.keys())
# Pour la reproductibilité, on fixe le random seed
random.seed(42)  
# On mange les speakers 
random.shuffle(speakers)

# Calculer le nombre de locuteurs pour le train
num_train_speakers = int(train_ratio * len(speakers))
train_speakers = speakers[:num_train_speakers]
test_speakers = speakers[num_train_speakers:]

print(f"Locuteurs pour le train ({len(train_speakers)}): {sorted(train_speakers)}")
print(f"Locuteurs pour le test ({len(test_speakers)}): {sorted(test_speakers)}")

# Vérification
assert len(set(train_speakers) & set(test_speakers)) == 0, \
    "ERREUR: Des locuteurs apparaissent dans train ET test!"

# On récupère les ficheirs pour le train et le test
train_list = []
test_list = []

for speaker in train_speakers:
    train_list.extend(speaker_files[speaker])

for speaker in test_speakers:
    test_list.extend(speaker_files[speaker])

print(f"\nNombre de fichiers dans train: {len(train_list)}")
print(f"Nombre de fichiers dans test: {len(test_list)}")
print(f"Ratio réel train/test: {len(train_list)/len(test_list):.2f}\n")

# Création des annotations en csv
train_csv_file = 'train_audioMNIST.csv'
with open(train_csv_file, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, lineterminator='\n')
    for item in train_list:
        csv_writer.writerow([item[0], item[1]])

test_csv_file = 'test_audioMNIST.csv'
with open(test_csv_file, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, lineterminator='\n')
    for item in test_list:
        csv_writer.writerow([item[0], item[1]])

