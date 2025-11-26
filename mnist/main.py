from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt  # 追加（冒頭）


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
            #入力チャネル数が1（MNISTの画像はモノクロなのでチャネル数は1）。
            #出力チャネル数が32（32個のフィルタを使って特徴マップを生成）。
            #カーネルサイズが3（3x3の畳み込みフィルタ）。
            #ストライドが1（畳み込みフィルタを1ピクセルずつ移動）。

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
            #入力チャネル数が32（前の畳み込み層からの出力）。
            #出力チャネル数が64（64個のフィルタを使用）。

        self.dropout1 = nn.Dropout2d(0.25) #ドロップアウト
        self.dropout2 = nn.Dropout2d(0.5)#ドロップアウト
        self.fc1 = nn.Linear(9216, 128)# 全結合層（Linear）
        self.fc2 = nn.Linear(128, 10)# 全結合層（Linear）

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)#活性化関数（ReLU）
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)#プーリング層（MaxPool2d）
        x = self.dropout1(x) 
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x) 
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) #出力　数字0〜9の分類を得る
        return output


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0  # 追加：train lossを合計する変数を追加
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # output（モデルの予測結果）とtarget（教師ラベル、つまり正解ラベル）を比較して損失を計算しています。
        train_loss += loss.item() * data.size(0)  # バッチ分のロスを加算する処理を追加
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)  # すべてのバッチの処理が終わったら、バッチごとの損失の合計を、全データの数でわって、平均にする部分を追加
    return train_loss  # 追加：train lossを返すための1行を追加


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() #テストデータに対しても同様に、F.nll_loss(output, target, reduction='sum')で損失を計算し、test_lossに加算しています。ここでも、output（モデルの予測出力）とtarget（教師ラベル）を比較して損失を算出しています。
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    
    return test_loss, accuracy  # 追加：lossと精度を返す


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device) #モデル（Net）に入力　→　各クラスの予測スコア（log_softmax）を得る。つまり数字0〜9の分類を得る。
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []  # 追加：train lossを記録
    test_losses = []   # 追加：test lossを記録
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
                #　先に定義されたtrainメソッド内においてloss = F.nll_loss(output, target)では損失関数（nll_Loss）でoutput（モデルの予測結果）とtarget（教師ラベル、つまり正解ラベル）と比較していると思われる　
                # 最適化（Adadelta）でパラメータ更新　　この二つが「学習」にあたる
                # バッチごとの損失の合計を、全データの数でわった平均lossを返す 
        test_loss, test_accuracy = test(args, model, device, test_loader) 
                #テストデータでは精度・lossを返す
        scheduler.step()
        
        # １エポック分が終わったら、リストに追加。
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)



    # 追加：lossをグラフにする
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')  # グラフ画像を保存
    plt.show()


    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
