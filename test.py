import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim


def activation(x):
    return 1 / (1 + torch.exp(x))


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


def ex1():
    torch.manual_seed(7)

    features = torch.randn((1, 3))

    n_input = features.shape[1]
    n_hidden = 2
    n_output = 1

    W1 = torch.randn((n_input, n_hidden))
    W2 = torch.randn((n_hidden, n_output))

    b1 = torch.randn((1, n_hidden))
    b2 = torch.randn((1, n_output))

    # y = activation(torch.mm(features, weights.view(5, 1)) + bias)
    y = activation(torch.mm(activation(torch.mm(features, W1) + b1), W2) + b2)

    print(y)

    a = np.random.rand(4, 3)
    b = torch.from_numpy(a)
    print(a)
    print(b)


class Network(nn.Module):
    def __init__(self):
        super.__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class Network1(nn.Module):
    def __init__(self):
        super.__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x


class Network2(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def ex2():
    # import helper
    import matplotlib.pyplot as plt
    print("ex2")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    traintset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traintset, batch_size=64, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)

    # inputs = images.view(images.shape[0], -1)
    # torch.manual_seed(7)
    # n_input = inputs.shape[1]
    # n_hidden = 256
    # n_output = 10
    # W1 = torch.randn((n_input, n_hidden))
    # W2 = torch.randn((n_hidden, n_output))
    # b1 = torch.randn((1, n_hidden))
    # b2 = torch.randn((1, n_output))
    # y = activation(torch.mm(activation(torch.mm(inputs, W1) + b1), W2) + b2)

    model = Network2()
    model.fc1.bias.data.fill_(0)
    model.fc1.weight.data.normal_(std=0.1)
    images.resize_(64, 1, 784)

    img_idx = 0
    ps = model.forward(images[img_idx, :])

    input_size = 728
    hidden_size = [128, 64]
    output_size = [10]

    model2 = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                           nn.ReLU(),
                           nn.Linear(hidden_size[0], hidden_size[1]),
                           nn.ReLU(),
                           nn.Linear(hidden_size[1], output_size),
                           nn.Softmax(dim=1))


def ex3():
    print("Hi3")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    traintset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traintset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")


def ex4():
    print("Hi4")
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(784, 256),
                          nn.ReLU(),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")


def ex5():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(784, 256),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in testloader:
                    images = images.view(images.shape[0], -1)
                    logits = model(images)
                    test_loss += criterion(logits, labels)
                    ps = torch.exp(logits)
                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            print(f'Test Accuracy: {accuracy.item() / len(testloader) * 100}%')
            print(f"Training loss: {running_loss / len(trainloader)}")
            print(f"Test loss: {test_loss / len(trainloader)}")


ex5()

