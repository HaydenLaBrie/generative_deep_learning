import torch
import matplotlib.pyplot as plt
from random import *
import numpy as np
from scipy.stats import norm
from vae_faces import VAE, Encoder, Decoder, testset


def reconstruct(model, testset, amount, device):
    fig = plt.figure(figsize=(15, 7))
    model.eval()
    b = randint(0, len(testset) - (amount - 1))
    with torch.no_grad():
        j = 1
        for i in range(amount):
            img = testset[i + b][0]
            pred = model(img.unsqueeze(0).to(device))
            pred = pred[2].squeeze(0).to('cpu')
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow((img.permute(1, 2, 0)))
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow((pred.permute(1, 2, 0)))
    plt.show()


def latent_space_distribution(model):
    # find a random spot in the testset and make sure it has the ability to have 128 total samples
    b = randint(0, len(testset) - (128 - 1))
    # each sample is a tensor size of 3 * 64 * 64
    # store samples in a list
    samples = []
    # make a loop to append all the 128 samples together
    for i in range(128):
        samples.append(testset[b + i][0])
    # translate the list to a 4d tensor
    samples = torch.stack(samples)
    # using model evaluation and no gradient updates
    model.eval()
    with torch.no_grad():
        # get the encoder z sample from distribution of median and variance
        # z will be 128 * 200 tensor
        z, _, _ = variational_autoencoder.encoder(samples, True)
    # create a numpy ndarray off 100 floats starting from -3 and ending at 3 with even spacing floats inbetween
    x = np.linspace(-3, 3, 100)
    # defining the single figures (aka window) width and height
    fig = plt.figure(figsize=(20, 5))
    # height space and width space between subplots as a ratio from total figures height/space
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    for i in range(50):
        # add a subplot at position i starting at 1
        # let the subplots be arranged as 5 rows and 10 columns
        ax = fig.add_subplot(5, 10, i + 1)
        # draw a diagram for each of the first 50 weights into 20 alloted bins and have a probability density function
        ax.hist(z[:, i], density=True, bins=20)
        ax.axis('off')
        ax.text(0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes)
        # plot standard normal distribution function of x
        ax.plot(x, norm.pdf(x))
    plt.show()


def new_faces(model, device):
    # going to have a 3 * 10 grid display of new images, thus 30 total images
    grid_width, grid_height = (10, 3)
    # get 30 samples of the latent space to draw from
    # thus we will get a ndarray of a list of 30 arrays of length 200 from random standard normalization sample
    # z_sample shape is (30,200), a batch of 30 z_samples
    z_sample = np.random.normal(size=(grid_width * grid_height, 200))
    # convert z_sample to tensor, as float32, on device
    # numpy is naturally float64 but model is float32 thus we must convert
    z_sample = torch.from_numpy(z_sample).to(torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        # from sampled latent space, get reconstructed images
        reconstructions = variational_autoencoder.decoder(z_sample)
    # define a figure of 18 width and 5 height
    fig = plt.figure(figsize=(18, 5))
    # height space and width space between subplots as a ratio from total figures height/space
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_width * grid_height):
        # create a new subplot
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        # get ith reconstruction image
        img = reconstructions[i]
        # change to 64 * 64 * 3 for matplotlib
        ax.imshow(img.permute(1, 2, 0))
    # plot all the images in the figure
    plt.show()


if __name__ == "__main__":
    # seeding to prevent randomization during different runs
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    device = "cpu"
    # initialize model
    encoder = Encoder(device).to(device)
    decoder = Decoder().to(device)
    variational_autoencoder = VAE(encoder, decoder).to(device)
    # load back trained weights
    state_dict = torch.load('variational_autoencoder.pth')
    variational_autoencoder.load_state_dict(state_dict)
    # # reconstruct images
    # reconstruct(variational_autoencoder, testset, 4, device)
    # # latent space distribution analysis
    # latent_space_distribution(variational_autoencoder)
    # sampling new faces from latent space
    new_faces(variational_autoencoder, device)
