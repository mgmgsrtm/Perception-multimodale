from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt   # Ajouté 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
            # Nombre de canaux d'entrée = 1 (images MNIST en niveaux de gris)
            # Nombre de canaux de sortie = 32 (32 filtres pour générer des cartes de caractéristiques)
            # Taille du noyau = 3 (filtre de convolution 3x3)
            # Stride = 1 (le filtre se déplace de 1 pixel à la fois)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
            # Nombre de canaux d'entrée = 32 (sortie de la couche précédente)
            # Nombre de canaux de sortie = 64 (64 filtres)

        self.dropout1 = nn.Dropout2d(0.25) # Dropout
        self.dropout2 = nn.Dropout2d(0.5)# Dropout
        self.fc1 = nn.Linear(9216, 128) #fully-connected layer
        self.fc2 = nn.Linear(128, 10) #fully-connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) # Fonction d'activation ReLU
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Couche de pooling MaxPool2d
        x = self.dropout1(x) 
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x) 
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # Sortie : classification des chiffres 0 à 
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0  # Ajouté : variable pour accumuler la perte d'entraînement
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # Compare la sortie du modèle (output) avec les labels corrects (target) pour calculer la perte
        train_loss += loss.item() * data.size(0) # Ajout de la perte totale pour ce batch
        loss.backward()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)  # Calcul de la perte moyenne sur tous les batches
    return train_loss   # Retourne la perte moyenne


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # Calcul de la perte sur les données de test et accumulation
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy   # Retourne la perte et la précision


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

    model = Net().to(device)  # Entrée dans le modèle Net → obtention des scores de prédiction pour chaque classe (0 à 9)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []   # Enregistrer la perte d'entraînement
    test_losses = []    # Enregistrer la perte de test
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
                # Dans la méthode train définie précédemment, loss = F.nll_loss(output, target) calcule la perte
                # en comparant la sortie du modèle (output) avec les labels corrects (target) via la fonction de perte nll_loss.
                # Les paramètres sont ensuite mis à jour avec l'optimiseur (Adadelta) ; ces deux étapes constituent l'apprentissage.
                # La perte moyenne renvoyée correspond à la somme des pertes par batch divisée par le nombre total d'échantillons.  
        test_loss, test_accuracy = test(args, model, device, test_loader) 
                # Pour les données de test, la précision et la perte sont renvoyées.
        scheduler.step()
        
        # Après chaque epoch, ajout aux listes
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)



     # Tracer la courbe de perte
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')   # Sauvegarde de l'image de la courbe# グラフ画像を保存
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
