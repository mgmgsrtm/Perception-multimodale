# With this code I have built a list with all the annotations.

import os

path = 'recordings/'
annot_list = []

#########################################
# Changement d'organisation des répertoires par rapport à l'auteur
# Nous avons un seul répertoire avec tous les fichiers sons à l'intérieur
# nous avons également que 3000 fichiers audio
########################################
for file in os.listdir(path):
    if file.lower().endswith('.wav'):   # check pour ne pas intégrer d'autres types de fichier
        file_path = path + file
        label = file[0]
        annot_list.append((file_path, label))

# Although the os.listdir does not follow any apparent order, I will shuffle it just to be safe.

import random
random.shuffle(annot_list)

# Split in train and test

train_size = int(0.75*len(annot_list))
train_list = annot_list[:train_size]
test_list = annot_list[train_size:]

# Finally created the annnotations file

import csv

with open('train_audioMNIST.csv', mode='w') as csv_file:  
    csv_writer = csv.writer(csv_file, lineterminator='\n')   # lineterminator='\n' => rend le code Windows compliant
    for item in train_list:
        csv_writer.writerow([item[0], item[1]])  

with open('test_audioMNIT.csv', mode='w') as csv_file:  
    csv_writer = csv.writer(csv_file, lineterminator='\n')   # lineterminator='\n' => rend le code Windows compliant
    for item in test_list:
        csv_writer.writerow([item[0], item[1]])