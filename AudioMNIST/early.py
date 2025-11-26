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

class MultimodaleDataset(Dataset):
    def __init__(self, annotations_file, train=True):
        # MNISTデータセットのロード
        self.mnist = datasets.MNIST('data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        
        # CSVファイルを使って音声データのパスとラベルを取得
        self.annotations = pd.read_csv(annotations_file, header=None, 
                                       names=['Path', 'Label'], delimiter=',')

        # 画像と音声データを格納するリスト
        self.tableau = []
        
        # MNISTデータセットと音声データを対応させる
        for i in range(len(self.mnist)):
            image, label = self.mnist[i]
            # 音声データをランダムに選択
            audio_paths = self.annotations[self.annotations['Label'] == label]['Path'].tolist() #Trueとなる行は音声ファイルがlabelに対応しているものをPythonのリストに変換
            audio_path = random.choice(audio_paths)
            self.tableau.append((image, label, audio_path))
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
        # MNISTの画像とラベル
        image, label = self.mnist[index]
        
        # 音声データのパス取得
        audio_path = self.tableau[index][2]
        
        # 音声データをスペクトログラムに変換
        pt_path = os.path.splitext(audio_path)[0] + '.pt'
        
        # スペクトログラムをロード、または計算
        if os.path.exists(pt_path):
            spectrogram = torch.load(pt_path, weights_only=True)
        else:
            audio_padded = torch.zeros((1, 48000))  # オーディオの長さに合わせて
            audio = torchaudio.load(audio_path)
            audio_padded[0, :len(audio[0])] = audio[0]  # 0チャンネルを使用
            spectrogram = self.ToSpectrogram(audio_padded)

        # この1行を入れないとMNIST のスケール差が 100〜1000倍
        # この行の意義　音声スペクトログラムの値の大きさを、画像(MNIST)と同じくらいのスケールに整えること
        # spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)


        # この時点で、[X, X, X, X] なのか[X, X, X,]なのか　ノートを見る
        # # この時点で、[X, X, X,]を[X, X, X, X] 変える必要があるらしい

        if spectrogram.size(1) != 28 or spectrogram.size(2) != 28:
            spectrogram_resized = F.interpolate(spectrogram.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
        else:
            spectrogram_resized = spectrogram    

        
        # 画像とスペクトログラムを水平方向に結合
        image_stucked = torch.cat((image, spectrogram_resized), dim=0)  # dim=0でスタック

        #ここで、ちゃんと[2,28,28]になっているか 
        
        return image_stucked, label
    

    def ToSpectrogram(self, audio):
        # スペクトログラムの計算（簡単な例）
        # ここではSTFT（短時間フーリエ変換）を使用
        return torchaudio.transforms.MelSpectrogram()(audio)

    # def ToDB(self, spectrogram):
    #     # dBスケールに変換
    #     return torchaudio.transforms.AmplitudeToDB()(spectrogram)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=10, kernel_size=7, stride=3),#要検討
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2),#要検討
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 2 * 2, 10)# なぜこの数字になったか
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
    print('Using {} device'.format(device)) #Use GPU if available

    model = CNN().to(device)
    print(model)

    model_loss = nn.CrossEntropyLoss()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = MultimodaleDataset(train=True, annotations_file='./train_audioMNIST.csv')

    # print("=== デバッグ：画像ラベル と 音声ファイル対応チェック ===")
    # for i in range(10):
    #     img, label, audio_path = train_dataset.tableau[i]
    #     print(label, audio_path)

    test_dataset = MultimodaleDataset(train=False, annotations_file='./test_audioMNIST.csv')



    # spectrogram, label = next(iter(train_dataset))
    # print(spectrogram.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(spectrogram[0], cmap='inferno')
    # plt.show()

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=10)

    # train_losses, train_scores, test_losses, test_scores = [], [], [], []
    # for epoch in range(30):
    #     print('Epoch:', epoch)
    #     train_loss, train_score = train_model(model, train_loader, model_loss, model_optimizer, device)
    #     test_loss, test_score = test_model(model, test_loader, model_loss, device)
    #     train_losses.append(train_loss)
    #     train_scores.append(train_score)
    #     test_losses.append(test_loss)
    #     test_scores.append(test_score)
    #     print(train_losses)
    #     print(train_scores)
    #     print(test_losses)
    #     print(test_scores)

    #   # Plot losses
    #     plt.plot(train_losses, label="Train Loss")
    #     plt.plot(test_losses, label="Test Loss")
    #     plt.legend()
    #     plt.title("Loss vs Epoch")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.show()

    #     # Plot accuracy
    #     plt.plot(train_scores, label="Train Accuracy")
    #     plt.plot(test_scores, label="Test Accuracy")
    #     plt.legend()
    #     plt.title("Accuracy vs Epoch")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.show()


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
    plt.title('Loss over Epochs Early fusion')

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
    

      