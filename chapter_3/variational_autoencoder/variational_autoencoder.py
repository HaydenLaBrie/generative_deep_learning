import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
from random import *

# fashion mnist is a 28x28 grayscale collection of images
# transform to 0-1 range of pixels and pad width and height by 2 on all sides,
# thus 4 total more pixels for width and 4 more for height
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Pad(2)])
batch_size = 100
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


# defining custom pytorch layer for sampling a random z from distribution
class Sampling(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, z_mean, z_log_var, training=False):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        # during training, we want to sample from z_log_var, thus get epsilon
        if training:
            # epsilon is a random sample from standard normal distribution
            # do this for 2x2 dimensions
            epsilon = torch.normal(mean=0.0, std=1.0, size=(batch, dim)).to(self.device)
        # during testing, we don't want variance to play a role
        else:
            # make epsilon 0 so that we don't have variance and just the mean
            epsilon = torch.zeros(batch, dim).to(self.device)
        # sigma of z_mean is exp(log(sigma^2)/2)
        # thus z_mean + sigma * (normal distribution sample) = z_sample
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


# compresses high-dimensional input data such as an image into a lower-dimensional embedding vector
# encodes to a latent space z
# embedding space and latent space are the same thing
class Encoder(nn.Module):
    def __init__(self, device):
        self.device = device
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
        self.fc2 = nn.Linear(2048, 2)
        # sample from distribution to get a z for decoder to use
        self.sample = Sampling(self.device)

    def forward(self, input_data, training=False):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        flatten = nn.Flatten()(x)
        z_mean = self.fc1(flatten)
        z_log_var = self.fc2(flatten)
        z = self.sample(z_mean, z_log_var, training)
        return z, z_mean, z_log_var


# decodes the embedding back to the original space
# takes in the z sample from the z_mean and z_log_var distribution
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
class VariationalAutoencoder(nn.Module):
    # initialize autoencoder with encoder and decoder objects
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_data, training=False):
        # first pass image to be decoded to latent space
        z, z_mean, z_log_var = self.encoder(input_data, training)
        # decode image from latent space back to pixel space
        reconstruction = self.decoder(z)
        # return not only reconstruction, but also z_mean and z_log_var
        # all 3 will be used in the loss function
        return z_mean, z_log_var, reconstruction


# custom loss function
# autoencoder only considered the reconstruction loss between copy and real image
# in variational autoencoder the reconstruction loss and kl loss is considered
def loss_fn(z_mean, z_log_var, reconstruction, input):
    # reconstruction loss is binary cross entropy between reconstruction and real image
    # doing beta VAE, a weighted reconstruction loss + kl loss
    # we do 500 * reconstruction loss here
    # beta = 500
    reconstruction_loss = torch.mean(
        500 * F.binary_cross_entropy(reconstruction, input)
    )
    # kl loss measures z_mean, z_log_var distribution difference to standard normal distribution
    kl_loss = torch.mean(
        torch.sum(-0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)), dim=1)
    )
    total_loss = reconstruction_loss + kl_loss
    return total_loss


def train(model, epochs, trainloader, device):
    # lists to store loss and accuracy
    loss_hist = [0] * epochs
    accuracy_hist = [0] * epochs
    for epoch in range(epochs):
        for x_batch, y_batch in trainloader:
            # put the batches on the device
            # only need x batch for autoencoder training
            x_batch = x_batch.to(device)
            # pass through encoder, then decode back and get result as prediction
            z_mean, z_log_var, reconstruction = model(x_batch, True)
            # loss is bce between individual pixels and the decoded reconstruction individual pixels
            loss = loss_fn(z_mean, z_log_var, reconstruction, x_batch)
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
            z_mean, z_log_var, reconstruction = model(x_batch, True)
            loss = loss_fn(z_mean, z_log_var, reconstruction, x_batch)
            loss_val += loss.item() * y_batch.size(0)
    # divide total loss by length of dataset to get avg loss
    loss_val /= len(testloader.dataset)
    print(f'Test set loss:{loss_val}')
    print('-' * 60)


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
    # initialize model
    encoder = Encoder(device).to(device)
    decoder = Decoder().to(device)
    variational_autoencoder = VariationalAutoencoder(encoder, decoder).to(device)
    learning_rate = 0.001
    summary(variational_autoencoder, (1, 32, 32))
    optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=learning_rate)
    epochs = 5
    # train
    train(variational_autoencoder, epochs, trainloader, device)
    # evaluate on test data
    test(variational_autoencoder, testloader, device)
    # save model
    torch.save(variational_autoencoder.state_dict(), 'variational_autoencoder.pth')
    print('Saved model as variational_autoencoder.pth')
    a = 1
    