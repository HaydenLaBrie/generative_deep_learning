import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
from random import *

# define a transformation that turnls PIL images to a torch.FloatTensor shape of channels * height * width in the
# range of [0,1]
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 32
# get the training dataset and apply the transformation to get it into a normalized tensor
# the type is a dataset
# dataset stores the samples and their corresponding labels
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# convert the dataset to a dataloader where the samples are shuffled and have a batch size
# num of workers: tells the data loader instance how many sub-processes to use for data loading
# theoretically, greater the num_workers, more efficiently the CPU load data and less the GPU has to wait.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# same thing is implemented for the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
# define a tuple which maps the y to the string
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, input_data):
        # take the input layer in flatten it
        flatten = nn.Flatten()(input_data)
        # 3072 to 200 fully connected layer
        x = self.fc1(flatten)
        # apply relu activation
        x = nn.ReLU()(x)
        # 200 to 150 fully connected layer
        x = self.fc2(x)
        # apply relu activation
        x = nn.ReLU()(x)
        # 150 to 10 fully connected layer
        x = self.fc3(x)
        return x


def train(model, num_epochs, trainloader, device):
    # lists to store loss and accuracy
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in trainloader:
            # put the batches on the device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # returns a (32,10) tensor. 10 possible label probabilities, for 32 predictions based on batch size
            pred = model(x_batch)
            # cross entropy loss for categorical label prediction
            loss = loss_fn(pred, y_batch)
            # back propagation
            loss.backward()
            # adam optimization to adjust weights
            optimizer.step()
            # zero out gradients for next batch
            optimizer.zero_grad()
            # add to total loss for epoch
            loss_hist[epoch] += loss.item() * y_batch.size(0)
            # add to accuracy history
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()
        # divide total loss by length of dataset to get avg loss
        loss_hist[epoch] /= len(trainloader.dataset)
        # divide accuracy sum by length of dataset to get avg accuracy
        accuracy_hist[epoch] /= len(trainloader.dataset)
        print(f'Epoch {epoch} loss:{loss_hist[epoch]} accuracy:{float(accuracy_hist[epoch])}')
        print('-' * 60)


def test(model, testloader, device):
    loss_val = 0
    accuracy = 0
    for x_batch, y_batch in testloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss_val += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy += is_correct.sum()
    # divide total loss by length of dataset to get avg loss
    loss_val /= len(testloader.dataset)
    # divide accuracy sum by length of dataset to get avg accuracy
    accuracy /= len(testloader.dataset)
    print(f'Test set loss:{loss_val} accuracy:{float(accuracy)}')
    print('-' * 60)


def analysis(model, testset, amount):
    b = randint(0, len(testset) - (amount - 1))
    fig = plt.figure(figsize=(15, 7))
    for i in range(amount):
        # (3, 32, 32) tensor of one image
        img = testset[i + b][0]
        label = classes[int(testset[i + b][1])]
        # get prediction of image
        # expects a batch dimension, thus add unsqueeze(0) to add 1 to batch size
        pred = model(img.unsqueeze(0).to(device))
        y_pred = classes[int(torch.argmax(pred))]
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(img.permute(1, 2, 0))
        ax.text(
            0.5, -0.25,
            f'Ground Truth: {label}\n Prediction: {y_pred}',
            size=16,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
    plt.show()


if __name__ == "__main__":
    # seeding to prevent randomization during different runs
    torch.manual_seed(1)
    # checking if cuda is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    # instantiating model
    mlp = Model().to(device)
    # printing summary for the given input size of the picture
    summary(mlp, (3, 32, 32))
    learning_rate = 0.0005
    # cross entropy for multiclass label prediction
    loss_fn = nn.CrossEntropyLoss()
    # use adam optimizer and apply give learning rate
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    # define epochs
    num_epochs = 10
    # train model
    train(mlp, num_epochs, trainloader, device)
    # evaluate on test data
    test(mlp, testloader, device)
    # analysis of test results
    analysis(mlp, testset, 10)
    a = 1
