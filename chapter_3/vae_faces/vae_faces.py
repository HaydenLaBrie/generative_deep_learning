import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

batch_size = 128

# celeba dataset is batch * 3 * 218 * 178
# it is mapped to a vector of 40 marking

# transform to tensor of 0-1 pixels, also resize image to 32 by 32 via bilinear interpolation
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Resize(size=(64, 64), antialias=True)])
transform = transforms.Compose([transforms.Resize(size=(64, 64), antialias=True),
                                transforms.ToTensor()])
trainset = torchvision.datasets.CelebA(root='./data', split='train',
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = torchvision.datasets.CelebA(root='./data', split='test',
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


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


class Encoder(nn.Module):
    def __init__(self, device):
        self.device = device
        super().__init__()
        # first convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            # batchnorm is done per channel
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # second convolution
        self.conv2 = nn.Sequential(
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # third convolution
        self.conv3 = nn.Sequential(
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # fourth convolution
        self.conv4 = nn.Sequential(
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # fifth convolution
        self.conv5 = nn.Sequential(
            # reducing the y*x dimension for each feature map by half
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # take flattened layer and map to 200 nodes to represent 200-dimensional latent space
        self.fc1 = nn.Linear(512, 200)
        self.fc2 = nn.Linear(512, 200)
        # sample from distribution to get a z for decoder to use
        self.sample = Sampling(self.device)

    def forward(self, input_data, training=False):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        flatten = nn.Flatten()(x)
        z_mean = self.fc1(flatten)
        z_log_var = self.fc2(flatten)
        z = self.sample(z_mean, z_log_var, training)
        return z, z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # take 2 nodes and fully connect to 2048 nodes
        self.fc1 = nn.Sequential(
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )
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
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # second transposed convolution
        self.tp2 = nn.Sequential(
            # take 128 channels and convert to 64 channels
            # here we get (64, 16, 16), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # third transposed convolution
        self.tp3 = nn.Sequential(
            # take 64 channels and convert to 32 channels
            # here we get (32, 32, 32), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # fourth transposed convolution
        self.tp4 = nn.Sequential(
            # take 64 channels and convert to 32 channels
            # here we get (32, 32, 32), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.tp5 = nn.Sequential(
            # take 64 channels and convert to 32 channels
            # here we get (32, 32, 32), thus we reduce channels by half and double channels width and length
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.fc1(input_data)
        x = torch.reshape(x, (-1, 128, 2, 2))
        x = self.tp1(x)
        x = self.tp2(x)
        x = self.tp3(x)
        x = self.tp4(x)
        x = self.tp5(x)
        return x


class VAE(nn.Module):
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


def loss_fn(z_mean, z_log_var, reconstruction, input):
    # reconstruction loss is binary cross entropy between reconstruction and real image
    # doing beta VAE, a weighted reconstruction loss + kl loss
    # we do 500 * reconstruction loss here
    # beta = 500
    reconstruction_loss = torch.mean(
        2000 * F.binary_cross_entropy(reconstruction, input)
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
    # # there are 162770 training images
    # print(len(trainset))
    # # each image is 3 * 218 * 178 => channels * height * width
    # # with bilinear interpolation we change each image to 3 * 32 * 32
    # print(trainset[0][0].size())
    # # each image has 40 binary target variables
    # print(trainset[0][1].size())
    # # with a batch of 128 there will be 1272 total batches
    # print(len(trainloader))
    # print('...')
    # # test set has 19962 test images of shape 3 * 64 * 32 with 40 binary targets
    # print(len(testset))
    # print(trainset[0][0].min())

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
    # summary(encoder, (3, 64, 64))
    decoder = Decoder().to(device)
    # summary(decoder, (1, 200))
    variational_autoencoder = VAE(encoder, decoder).to(device)
    summary(variational_autoencoder, (3, 64, 64))
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=learning_rate)
    epochs = 10
    # train
    train(variational_autoencoder, epochs, trainloader, device)
    # evaluate on test data
    test(variational_autoencoder, testloader, device)
    # save model
    torch.save(variational_autoencoder.state_dict(), 'variational_autoencoder.pth')
    print('Saved model as variational_autoencoder.pth')
    a = 1
