import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
from random import *

# fashion mnist is a 28x28 grayscale collection of images
# transform to 0-1 range of pixels and pad width and height by 2 on all sides,
# thus 4 total more pixels for width and 4 more for height
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Pad(2)])
batch_size = 32
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


# compresses high-dimensional input data such as an image into a lower-dimensional embedding vector
# encodes to a latent space z
# embedding space and latent space are the same thing
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # first convolution
        self.conv1 = nn.Sequential(
            # taking a grayscale channel and converting it to 32 features
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU()
        )
        # second convolution
        self.conv2 = nn.Sequential(
            # taking 32 feature maps and creating 64 feature maps out of it
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU()
        )
        # third convolution
        self.conv3 = nn.Sequential(
            # taking 64 feature maps and creating 128 feature maps out of it
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU()
        )
        # take flattened layer and map to 2 nodes to represent 2-dimensional latent space
        self.fc1 = nn.Linear(2048, 2)

    def forward(self, input_data):
        x = self.conv1(input_data)
        print(torch.max(x))
        x = self.conv2(x)
        x = self.conv3(x)
        print(torch.max(x))
        print('.......')
        flatten = nn.Flatten()(x)
        x = self.fc1(flatten)
        return x


# decompresses a given embedding vector back to the original domain (e.g., back to an image)
# takes the latent space z and decodes it back to the original pixel space
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # take 2 nodes and fully connect to 2048 nodes
        self.fc1 = nn.Linear(2, 2048)
        # apply a reshape from a vector of 2048 to (128,4,4)
        # first transposed convolution
        self.tp1 = nn.Sequential(
            # take 128 channels and convert to 128 channels
            # here we get (128, 8, 8,), thus expanding channels my double
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU()
        )
        # second transposed convolution
        self.tp2 = nn.Sequential(
            # take 128 channels and convert to 64 channels
            # here we get (64, 16, 16), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU()
        )
        # third transposed convolution
        self.tp3 = nn.Sequential(
            # take 64 channels and convert to 32 channels
            # here we get (32, 32, 32), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU()
        )
        # apply convolution
        # take 32 channels and convert to 1 channel (meant to represent the gray channel)
        # convolution will give (1, 32, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=1,
                kernel_size=3, stride=1,
                padding=1
            ),
            # apply sigmoid activation to get a 0-1 pixel range
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.fc1(input_data)
        x = torch.reshape(x, (-1, 128, 4, 4))
        x = self.tp1(x)
        x = self.tp2(x)
        x = self.tp3(x)
        x = self.conv1(x)
        return x


# combining encoder and decoder to make autoencoder
class Autoencoder(nn.Module):
    # initialize autoencoder with encoder and decoder objects
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_data):
        # first pass image to be decoded to latent space
        x = self.encoder(input_data)
        # decode image from latent space back to pixel space
        x = self.decoder(x)
        return x


def train(model, num_epochs, trainloader, device):
    # lists to store loss and accuracy
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in trainloader:
            # put the batches on the device
            # only need x batch for autoencoder training
            x_batch = x_batch.to(device)
            # pass through encoder, then decode back and get result as prediction
            pred = model(x_batch)
            # loss is bce between individual pixels and the decoded reconstruction individual pixels
            loss = loss_fn(pred, x_batch)
            # back propagation
            loss.backward()
            # adam optimization to adjust weights
            optimizer.step()
            # zero out gradients for next batch
            optimizer.zero_grad()
            # add to total loss for epoch
            # loss.item() finds loss average of the total batch
            # x_batch.size(0) gets the amount of samples in a batch
            # thus we multiply both to demonstrate avg total loss sum
            loss_hist[epoch] += loss.item() * x_batch.size(0)
        # divide total loss by length of dataset to get avg loss
        loss_hist[epoch] /= len(trainloader.dataset)
        print(f'Epoch {epoch} loss:{loss_hist[epoch]}')
        print('-' * 60)


def test(model, testloader, device):
    model.eval()
    loss_val = 0
    # toggle gradient calculation off
    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, x_batch)
            loss_val += loss.item() * y_batch.size(0)
    # divide total loss by length of dataset to get avg loss
    loss_val /= len(testloader.dataset)
    print(f'Test set loss:{loss_val}')
    print('-' * 60)


def reconstruct(model, testset, amount, device):
    fig = plt.figure(figsize=(15, 7))
    model.eval()
    b = randint(0, len(testset) - (amount - 1))
    with torch.no_grad():
        j = 1
        for i in range(amount):
            img = testset[i + b][0]
            pred = model(img.unsqueeze(0).to(device))
            pred = pred.squeeze(0).to('cpu')
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow(img.permute(1, 2, 0), cmap='gray')
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow(pred.permute(1, 2, 0), cmap='gray')
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
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = Autoencoder(encoder, decoder).to(device)
    summary(autoencoder, (1, 32, 32))
    # define loss function
    loss_fn = nn.BCELoss()
    # define optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    batch_size = 100
    num_epochs = 5
    # train model
    train(autoencoder, num_epochs, trainloader, device)
    # evaluate on test data
    test(autoencoder, testloader, device)
    # reconstruct images
    reconstruct(autoencoder, testset, 4, device)
    # save model and then load back in, so I don't have to train everytime
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    print('Saved model as autoencoder.pth')
    a = 1
