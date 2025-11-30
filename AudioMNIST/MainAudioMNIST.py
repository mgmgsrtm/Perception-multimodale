import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn #API for building neural networks
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes
import os
import torchaudio
import matplotlib.pyplot as plt

class AudioDatasetSpectrogram(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        self.ToSpectrogram = torchaudio.transforms.MelSpectrogram()
        self.ToDB = torchaudio.transforms.AmplitudeToDB()
        
    def __len__(self):
        return(len(self.annotations))
    
    def __getitem__(self, index):
        # Récupérer le chemin du fichier audio et le label
        audio_path = self.annotations.iloc[index, 0]
        label = self.annotations.iloc[index, 1]
        
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
            spectrogram = self.ToDB(spectrogram)
            
            # Sauvegarder le spectrogramme dans un fichier .pt
            torch.save(spectrogram, pt_path)
        
        return spectrogram, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(38*19*10, 10)
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

    print(torch.backends.mps.is_available())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using", device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN().to(device)
    print(model)

    model_loss = nn.CrossEntropyLoss()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = AudioDatasetSpectrogram('./train_audioMNIST.csv')
    test_dataset = AudioDatasetSpectrogram('./test_audioMNIST.csv')


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
        
    # Tracer le graphique
    plt.figure(figsize=(12, 6))

    # Tracer la perte
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(30), train_losses, label='Train Loss')
    plt.plot(np.arange(30), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Tracer la précision
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(30), train_scores, label='Train Accuracy')
    plt.plot(np.arange(30), test_scores, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Afficher le graphique
    plt.tight_layout()
    plt.savefig('loss_accuracy_curve.png')  # Sauvegarde de l'image de la courbe
    plt.show()


    

if __name__ == '__main__':
    main()
    

      