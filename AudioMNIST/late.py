import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn #API for building neural networks
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes



class MultiModalDataset(Dataset):
    def __init__(self, train=True):
    # Chargement du dataset MNIST
        self.mnist = datasets.MNIST('data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        
        # Tableau contenant des tuples (label, filepath) pour AudioMNIST
        self.tableau = []

        for digit in range(10): # Traiter chaque chiffre de 0 à 9
            folder = "recordings" # Dossier contenant les fichiers audio d'AudioMNIST
            for foldername in os.listdir(folder):
                # Sélectionner uniquement les fichiers dont le nom commence par le chiffre correspondant et se termine par ".wav"
                # → Cela extrait uniquement les fichiers audio correspondant à l'étiquette du chiffre
                if foldername.startswith(f"{digit}_") and foldername.endswith(".wav"):
                    self.tableau.append((digit, os.path.join(folder, foldername))) # Ajouter le tuple (label, filepath) à self.tableau
        
        mnist_size = len(self.mnist)
        audio_size = len(self.tableau)
        
        # Ajuster la taille en fonction du plus petit dataset si MNIST et AudioMNIST n'ont pas le même nombre d'échantillons
        self.size = min(mnist_size, audio_size)

        self.ToSpectrogram = torchaudio.transforms.MelSpectrogram()

        
    def __len__(self):
        return self.size   # Retourne la taille du dataset
    
    def __getitem__(self, index):

         # Vérifier si l'indice est dans la plage
        if index >= self.size:
            raise IndexError(f"Index {index} out of range for dataset with size {self.size}")
       
       # Image et label MNIST
        image, label = self.mnist[index]

        # Récupérer le chemin du fichier audio
        audio_path = self.tableau[index][1]
        
        # Construire le chemin du fichier .pt correspondant
        pt_path = os.path.splitext(audio_path)[0] + '.pt'
        
        # Vérifier si le fichier .pt existe déjà
        if os.path.exists(pt_path):
            # Charger le spectrogramme depuis le fichier .pt
            spectrogram = torch.load(pt_path, weights_only=True)
        else:
            # Calculer le spectrogramme depuis le fichier audio
            audio_padded = torch.zeros((1, 48000))
            audio = torchaudio.load(audio_path)
            audio_padded[0, :len(audio[0][0])] = audio[0][0]
            spectrogram = self.ToSpectrogram(audio_padded)

        return image, spectrogram, label

# Définition de conv2d, input shape = (N: taille du batch, C, H, W)

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7220, 10) 
        )
        
    def forward(self, x):
        logits = self.ConvNet(x)
        return logits
    
    # def forward(self, x):
    #     print("AudioCNN input shape:", x.shape)  # 入力形状をプリント
    #     x = self.ConvNet[0](x)  # 最初のConv2d
    #     print("After 1st Conv2d:", x.shape)  # 1回目のConv2d後の形状
    #     x = self.ConvNet[2](x)  # 2回目のConv2d
    #     print("After 2nd Conv2d:", x.shape)  # 2回目のConv2d後の形状
    #     x = self.ConvNet[4](x)  # Flatten
    #     print("After Flatten:", x.shape)  # Flatten後の形状
    #     x = self.ConvNet[5](x)  # Linear
    #     print("After Linear:", x.shape)  # Linear後の形状
    #     return x

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40, 10) 
        )

    def forward(self, x):
        logits = self.ConvNet(x)
        return logits

    # def forward(self, x):
    #     print("ImageCNN input shape:", x.shape)
    #     x = self.ConvNet[0](x)  # 最初のConv2d
    #     print("After 1st Conv2d:", x.shape)
    #     x = self.ConvNet[2](x)  # 2回目のConv2d
    #     print("After 2nd Conv2d:", x.shape)
    #     x = self.ConvNet[4](x)  # Flatten
    #     print("After Flatten:", x.shape)
    #     x = self.fc(x)  # 全結合層を直接適用
    #     print("After Linear:", x.shape)
    #     return x

    # def forward(self, x):
    #     print("AudioCNN input shape:", x.shape)  # 入力形状をプリント
    #     x = self.ConvNet[0](x)  # 最初のConv2d
    #     print("After 1st Conv2d:", x.shape)  # 1回目のConv2d後の形状
    #     x = self.ConvNet[2](x)  # 2回目のConv2d
    #     print("After 2nd Conv2d:", x.shape)  # 2回目のConv2d後の形状
    #     x = self.ConvNet[4](x)  # Flatten
    #     print("After Flatten:", x.shape)  # Flatten後の形状
    #     x = self.ConvNet[5](x)  # Linear
    #     print("After Linear:", x.shape)  # Linear後の形状
    #     return x

    

class LateFusionCNN(nn.Module):
    def __init__(self):
        super(LateFusionCNN, self).__init__()

        self.image_cnn = ImageCNN()
        self.audio_cnn = AudioCNN()

        # Couche recevant les caractéristiques fusionnées
        self.ConvNet = nn.Sequential(
            nn.Linear(20, 64),  # Dimensions d'entrée = 20 (10 features image + 10 features audio)
            nn.ReLU(),
            nn.Linear(64, 10)  # Conversion finale en 10 classes
        )
    
    def forward(self, image, audio):
        # Extraire les caractéristiques avec les CNN
        img_feat = self.image_cnn(image)
        aud_feat = self.audio_cnn(audio)

        # Fusion tardive des caractéristiques
        fusion = torch.cat((img_feat, aud_feat), dim=1)

        # Classification via couches fully-connected
        logits = self.ConvNet(fusion)  # 最終的にクラスを出力
        return logits






    

def train_model(model, dataloader, model_loss, optimizer, device, mean=0):
    total_loss = 0
    total_correct = 0
    model.train() # Specifies that the model should compute gradients
    for images, spectrograms, labels in dataloader:  # Traiter le batch de 50 échantillons
        images = images.to(device)
        spectrograms = spectrograms.to(device)-mean
        labels = labels.to(device)
        # Forward pass
        prediction = model(images, spectrograms)# Traitement de 50 échantillons
        loss = model_loss(prediction, labels.long()) # Perte moyenne sur le batch
        # Update loss and score
        total_loss += loss.item() # Addition de la perte de tous les batches dans l'epoch
        total_correct += (prediction.argmax(1)==labels).sum().item()
        # Backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    size = len(dataloader.dataset)
    avg_loss = total_loss/size
    score = total_correct/size
    return(avg_loss, score)

def test_model(model, dataloader, model_loss, device, mean=0):
    total_loss = 0
    total_correct = 0
    model.eval() #Specifies that the model does not need to keep track of gradients
    for images, spectrograms, labels in dataloader:
        images = images.to(device)
        spectrograms = spectrograms.to(device)-mean
        labels = labels.to(device)
        prediction = model(images, spectrograms)
        loss = model_loss(prediction, labels.long())
        total_loss += loss.item()
        total_correct += (prediction.argmax(1) == labels).sum().item()
        
    size = len(dataloader.dataset)
    avg_loss = total_loss/size
    score = total_correct/size
    return(avg_loss, score)






def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LateFusionCNN().to(device)
    print(model) # Initialiser le modèle LateFusionCNN et le transférer sur le device

    # Définir la fonction de perte
    model_loss = nn.CrossEntropyLoss()
    # Définir l'optimiseur SGD avec un learning rate de 0.001 et un momentum de 0.9
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Créer les datasets multimodaux pour l'entraînement et le test
    train_dataset = MultiModalDataset('./train_audioMNIST.csv')
    test_dataset = MultiModalDataset('./test_audioMNIST.csv')

    # Créer le DataLoader 
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=10) #num_workers=10とは
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=10)

    # Listes pour stocker les pertes et précisions à chaque epoch 
    train_losses, train_scores, test_losses, test_scores = [], [], [], []

    for epoch in range(30):
        print('Epoch:', epoch)
        train_loss, train_score = train_model(model, train_loader, model_loss, model_optimizer, device)
        test_loss, test_score = test_model(model, test_loader, model_loss, device)
        train_losses.append(train_loss)
        train_scores.append(train_score)
        test_losses.append(test_loss)
        test_scores.append(test_score)
        print(train_losses)
        print(train_scores)
        print(test_losses)
        print(test_scores)
        
    # Tracer le graphique
    plt.figure(figsize=(12, 6))

    # Tracer la perte
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(30), train_losses, label='Train Loss')
    plt.plot(np.arange(30), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs Late fusion')

    # Tracer accuracy
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(30), train_scores, label='Train Accuracy')
    plt.plot(np.arange(30), test_scores, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Afficher le graphique
    plt.tight_layout()
    plt.savefig('lateloss_accuracy_curve.png')  # Sauvegarde de l'image de la courbe
    plt.show()

if __name__ == '__main__':
    main()
