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


#あとで名前MultiModalDataset変更
class MultiModalDataset(Dataset):
    def __init__(self, annotations_file, train=True):
    # MNISTデータセットのロード
        self.mnist = datasets.MNIST('data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        
        # AudioMNIST のファイルテーブル(label, label, filepath) のタプルが複数入る
        self.tableau = []

        for digit in range(10):
            folder = "recordings"
            for foldername in os.listdir(folder):
                if foldername.startswith(f"{digit}_") and foldername.endswith(".wav"):
                    self.tableau.append((digit, os.path.join(folder, foldername))) #(ラベル, filepath) というタプルを self.tableau に追加
        
   
        self.ToSpectrogram = torchaudio.transforms.MelSpectrogram()

        
    def __len__(self):
        return(len(self.mnist))
    
    def __getitem__(self, index):
        # MNISTの画像とラベル
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




#conv2dの定義は、input shape = (N:taille de badge, C, H, W)

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=3), #要検討（入力データのチャンネル数、畳み込みで学習するフィルタの数、カーネルの大きさ）
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2), #要検討
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(38*19*10, 10) #要確認
        )
        
    def forward(self, x):
        logits = self.ConvNet(x)
        return(logits)
    

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.ConvNet = nn.Sequential(   
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=3), #要検討
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2), #要検討
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(38*19*10, 10) #要確認
        )
        
    def forward(self, x):
        logits = self.ConvNet(x)
        return(logits)
    

class LateFusionCNN(nn.Module):
    def __init__(self):
        super(LateFusionCNN, self).__init__()

        self.image_cnn = ImageCNN()
        self.audio_cnn = AudioCNN()

        # 画像も音声も最終的に32*4*4=512 の特徴にする設計 で返してきたが、
        # 上のImageCNN　AudioCNNでは　それぞれ10次元
        self.ConvNet = nn.Sequential(
            nn.Linear(10 + 10, 64), #なに？　結合された 20 次元（10+10）を受け取り、より高次の 64 次元特徴に変換する層
            nn.ReLU(),                  #なにをしている？非線形変換（活性化関数) 0か1以上
            nn.Linear(64, 10)          #なにをしている？最終的に 10 クラス（数字 0〜9）のロジットに変換
        )
    
    def forward(self, image, audio):
        # CNN で別々に特徴抽出
        img_feat = self.image_cnn(image)
        aud_feat = self.audio_cnn(audio)

        # 結合（late fusion）
        fusion = torch.cat((img_feat, aud_feat), dim=1)

        # 全結合で分類
        logits = self.ConvNet(fusion) #最終的なクラスをここで返す
        return logits
    


    

def train_model(model, dataloader, model_loss, optimizer, device, mean=0):
    total_loss = 0
    total_correct = 0
    model.train() # Specifies that the model should compute gradients
    for images, spectrograms, labels in dataloader:  #バッジの中の50のデータを処理
        #Zero mean and transfer data to device
        images = images.to(device)
        spectrograms = spectrograms.to(device)-mean
        labels = labels.to(device)
        # Forward pass
        prediction = model(images, spectrograms)# 50個を一気に処理
        loss = model_loss(prediction, labels.long()) #バッジの平均ロス
        # Update loss and score
        total_loss += loss.item() # 一度のエポック内で全てのバッジの平均を足していく
        total_correct += (prediction.argmax(1)==labels).sum().item()
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
    print('Using {} device'.format(device)) #Use GPU if available

    model = LateFusionCNN().to(device)
    print(model)

    model_loss = nn.CrossEntropyLoss()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = MultiModalDataset('./train_audioMNIST.csv')
    test_dataset = MultiModalDataset('./test_audioMNIST.csv')

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=10) #num_workers=10とは
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
        
    # グラフ化
    plt.figure(figsize=(12, 6))

    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(30), train_losses, label='Train Loss')
    plt.plot(np.arange(30), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs Late fusion')

    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(30), train_scores, label='Train Accuracy')
    plt.plot(np.arange(30), test_scores, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # グラフを表示
    plt.tight_layout()
    plt.savefig('loss_accuracy_curve.png')  # グラフ画像を保存
    plt.show()

if __name__ == '__main__':
    main()
    


#問題

#MelSpectrogram の出力 size と Linear(381910) が一致しない

#torchaudio.load(audio_path) の返り値の扱いが誤っている

#AudioMNIST の件数と MNIST の件数が一致しない

#num_workers=10 は危険 らしい