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
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=2)
# define a tuple which maps the y to the string
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            # take a (3,32,32) to a (32,32,32), take the 3 input channels and derive 32 features
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=3, padding='same',
                stride=1
            ),
            # apply batch normalization to each channel to prevent exploding gradients
            nn.BatchNorm2d(num_features=32),
            # apply activation to each cell across all channels
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            # take a (32,32,32) to a (32,16,16), take the 32 input channels and derive 32 features
            # the stride of 2 reduces the height and width by half 32/2=16
            nn.Conv2d(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=1
            ),
            # apply batch normalization to each channel to prevent exploding gradients
            nn.BatchNorm2d(num_features=32),
            # apply activation to each cell across all channels
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            # take a (32,16,16) to a (64,16,16), take the 32 input channels and derive 64 features
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            ),
            # apply batch normalization to each channel to prevent exploding gradients
            nn.BatchNorm2d(num_features=64),
            # apply activation to each cell across all channels
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            # take a (64,16,16) to a (64,8,8), take the 64 input channels and derive 64 features
            # the stride of 2 reduces the height and width by half 16/2=8
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=2, padding=1
            ),
            # apply batch normalization to each channel to prevent exploding gradients
            nn.BatchNorm2d(num_features=64),
            # apply activation to each cell across all channels
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            # this will take the flatten result of the convolutions which is 4096 cells, and map them to 128 nodes
            nn.Linear(4096, 128),
            # apply batch normalization to each channel to prevent exploding gradients
            nn.BatchNorm1d(num_features=128),
            # apply activation to each cell across all channels
            nn.LeakyReLU(),
            # randomly turn off half of the cells, a.k.a make them 0
            # this will be used to send to the final fully connected layer
            nn.Dropout(p=0.5)
        )
        # take the 128 nodes with the drop out of half and connect to 10 nodes for classification
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        flatten = nn.Flatten()(x)
        x = self.fc1(flatten)
        x = self.fc2(x)
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
    model.eval()
    loss_val = 0
    accuracy = 0
    # toggle gradient calculation off
    with torch.no_grad():
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
    # call eval to use running estimates of batch norm and turn off drop
    model.eval()
    b = randint(0, len(testset) - (amount - 1))
    fig = plt.figure(figsize=(15, 7))
    # toggle gradient calculation off
    with torch.no_grad():
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
    torch.backends.cudnn.deterministic = True
    # checking if cuda is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    cnn = Model().to(device)
    # printing summary for the given input size of the picture
    summary(cnn, (3, 32, 32))
    learning_rate = 0.0005
    # cross entropy for multiclass label prediction
    loss_fn = nn.CrossEntropyLoss()
    # use adam optimizer and apply give learning rate
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    # define epochs
    num_epochs = 10
    # train model
    train(cnn, num_epochs, trainloader, device)
    # evaluate on test data
    test(cnn, testloader, device)
    # analysis of test results
    analysis(cnn, testset, 10)
    a = 1
