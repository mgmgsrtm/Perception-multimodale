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
from torch import nn 
from torch.utils.data import Dataset, DataLoader 

class MultimodaleDataset(Dataset):
    def __init__(self, annotations_file, train=True):
        # Chargement du dataset MNIST
        self.mnist = datasets.MNIST('data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        
        # Charger les chemins et labels des fichiers audio à partir d'un fichier CSV
        self.annotations = pd.read_csv(annotations_file, header=None, 
                                       names=['Path', 'Label'], delimiter=',')

        # Liste pour stocker les données images et audio
        self.tableau = []
        
        # Associer chaque image MNIST à un fichier audio
        for i in range(len(self.mnist)):
            image, label = self.mnist[i]
            # Sélection aléatoire d'un fichier audio correspondant au label
            # Convertir en liste les chemins audio dont le label correspond
            audio_paths = self.annotations[self.annotations['Label'] == label]['Path'].tolist()
            audio_path = random.choice(audio_paths)
            self.tableau.append((image, label, audio_path))
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
        # Récupérer l'image et le label MNIST
        image, label = self.mnist[index]
        
        # Récupérer le chemin du fichier audio
        audio_path = self.tableau[index][2]
        
         # Convertir le fichier audio en spectrogramme
        pt_path = os.path.splitext(audio_path)[0] + '.pt'
        
         # Charger le spectrogramme depuis le fichier .pt, sinon le calculer
        if os.path.exists(pt_path):
            spectrogram = torch.load(pt_path, weights_only=True)
        else:
            audio_padded = torch.zeros((1, 48000))  # Ajuster à la longueur de l'audio
            audio = torchaudio.load(audio_path)
            audio_padded[0, :len(audio[0])] = audio[0]  # Utiliser le canal 0
            spectrogram = self.ToSpectrogram(audio_padded)

        # Normaliser le spectrogramme pour avoir une échelle comparable à MNIST
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

        # Redimensionner le spectrogramme si nécessaire pour correspondre à 28x28
        if spectrogram.size(1) != 28 or spectrogram.size(2) != 28:
            spectrogram_resized = F.interpolate(spectrogram.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
        else:
            spectrogram_resized = spectrogram    

        
        # Empiler horizontalement l'image et le spectrogramme
        image_stucked = torch.cat((image, spectrogram_resized), dim=0)  # empilement sur dim=0

        return image_stucked, label
    

    def ToSpectrogram(self, audio):
        # Calcul du spectrogramme à partir de l'audio
        # Ici, utilisation de la STFT (Transformée de Fourier à court terme)
        return torchaudio.transforms.MelSpectrogram()(audio)

    # def ToDB(self, spectrogram):
    #     # dBスケールに変換
    #     return torchaudio.transforms.AmplitudeToDB()(spectrogram)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=10, kernel_size=7, stride=3),#
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2),#
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 2 * 2, 10)
        )
        
    def forward(self, x):
        logits = self.ConvNet(x)
        return(logits)


    

def train_model(model, dataloader, model_loss, optimizer, device, mean=0):
    total_loss = 0
    total_correct = 0
    model.train() # Specifies that the model should compute gradients
    for X, y in dataloader:
        #Zero mean and transfer data to device
        X = X.to(device)
        X = X - mean
        y = y.to(device)
        # Forward pass
        prediction = model(X)
        loss = model_loss(prediction, y.long())
        # Update loss and score
        total_loss += loss.item()
        total_correct += (prediction.argmax(1)==y).sum().item()
        # Backward pass
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
    for X, y in dataloader:
        X = X.to(device)
        X = X - mean
        y = y.to(device)
        prediction = model(X)
        loss = model_loss(prediction, y.long())
        total_loss += loss.item()
        total_correct += (prediction.argmax(1) == y).sum().item()
        
    size = len(dataloader.dataset)
    avg_loss = total_loss/size
    score = total_correct/size
    return(avg_loss, score)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialiser le modèle et le transférer sur le device
    model = CNN().to(device)
    print(model)

    model_loss = nn.CrossEntropyLoss()

    # Définir l'optimiseur SGD avec learning rate et momentum
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Créer le dataset d'entraînement multimodal
    train_dataset = MultimodaleDataset(train=True, annotations_file='./train_audioMNIST.csv')

    # print("----- Vérification paire -------")
    # for i in range(10):
    #     img, label, audio_path = train_dataset.tableau[i]
    #     print(label, audio_path)

    # Créer le dataset de test multimodal
    test_dataset = MultimodaleDataset(train=False, annotations_file='./test_audioMNIST.csv')

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=10)

   


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
        
    # Visualisation
    plt.figure(figsize=(12, 6))

     # Tracé des pertes
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(30), train_losses, label='Train Loss')
    plt.plot(np.arange(30), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs Early fusion')

    # Tracé des précisions
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(30), train_scores, label='Train Accuracy')
    plt.plot(np.arange(30), test_scores, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Affichage du graphique
    plt.tight_layout()
    plt.savefig('earlyloss_accuracy_curve.png')  # Sauvegarder le graphique
    plt.show()


    

if __name__ == '__main__':
    main()
    

      