import torch
import matplotlib.pyplot as plt
from random import *
from variational_autoencoder import VariationalAutoencoder, Encoder, Decoder, testset, testloader
import numpy as np


def reconstruct(model, testset, amount, device):
    fig = plt.figure(figsize=(15, 7))
    model.eval()
    b = randint(0, len(testset) - (amount - 1))
    with torch.no_grad():
        j = 1
        for i in range(amount):
            img = testset[i + b][0]
            z_mean, z_log_var, reconstruction = model(img.unsqueeze(0).to(device))
            pred = reconstruction.squeeze(0).to('cpu')
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow(img.permute(1, 2, 0), cmap='gray')
            ax = fig.add_subplot(2, 5, j)
            j = j + 1
            ax.imshow(pred.permute(1, 2, 0), cmap='gray')
    plt.show()


def get_z_mean(model, testloader, device):
    # define a list to hold embeddings
    embeddings = []
    # define a list to hold colors
    colors = []
    # evaluating model
    model.eval()
    # dont calculate gradients
    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_mean, z_log_var, reconstruction = model.encoder(x_batch, device)
            # convert the tensor to a list
            pred = z_mean.tolist()
            y_batch = y_batch.tolist()
            # add list to list
            embeddings.extend(pred)
            colors.extend(y_batch)
            # do the first 5000
            if len(embeddings) >= 5000:
                break
            # convert to numpy arrays
        embeddings = np.array(embeddings)
        colors = np.array(colors)
    return embeddings, colors


def latent_space_show(embeddings, colors, samples):
    # convert tensor to numpy
    samples = samples.numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap="rainbow",
        c=colors,
        alpha=0.8,
        s=3
    )
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        marker='o',
        c='blue'
    )
    plt.colorbar()


def generate(images, samples):
    # force numpy values to only show 2 decimal points
    np.set_printoptions(precision=2)
    # figure size
    fig = plt.figure(figsize=(15, 7))
    # convert tensor to numpy
    samples = samples.numpy()
    for i in range(images.size(0)):
        # get ith encoding embedding
        encoding = samples[i]
        img = images[i]
        # 2 x 9 image matrix, plot at i+1 during i loop
        ax = fig.add_subplot(2, 9, i + 1)
        # show image and have 1,2 be the x and y and dimension 0 be the batch
        ax.imshow(img.permute(1, 2, 0), cmap='gray')
        # display embeddings below generated images
        ax.text(0.5, -0.3, f'{encoding}', size=10, color='blue',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)


if __name__ == "__main__":
    # seeding to prevent randomization during different runs
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    device = "cpu"
    # initialize model
    encoder = Encoder(device).to(device)
    decoder = Decoder().to(device)
    variational_autoencoder = VariationalAutoencoder(encoder, decoder).to(device)
    # load back trained weights
    state_dict = torch.load('variational_autoencoder.pth')
    variational_autoencoder.load_state_dict(state_dict)
    # get the embeddings of all the test samples
    # embeddings are a 2 ndim array
    # colors are a 1 ndim array
    # embeddings = z_mean = [x, y] => C = [c]
    embeddings, colors = get_z_mean(variational_autoencoder, testloader, device)
    # sample from standard normal distribution
    sample = np.random.normal(size=(18, 2))
    sample = torch.from_numpy(sample).to(torch.float32)
    # print(type(sample))
    variational_autoencoder.decoder.eval()
    # put the latent space/embeddings into the decoder to create images
    with torch.no_grad():
        reconstructions = variational_autoencoder.decoder(sample)
    # show images that the decoder created
    generate(reconstructions, sample)
    # plot latent space
    latent_space_show(embeddings, colors, sample)
    plt.show()
    a = 1
    