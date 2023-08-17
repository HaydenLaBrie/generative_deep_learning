import torch
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder, Encoder, Decoder, testloader
from analysis import get_embeddings, plot_color_latent_space


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


if __name__ == "__main__":
    device = "cpu"
    # uncomment next line to consistently have same run:
    # np.random.seed(1)
    # load back the model
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = Autoencoder(encoder, decoder).to(device)
    state_dict = torch.load('autoencoder.pth')
    autoencoder.load_state_dict(state_dict)
    # get the embeddings of all the test samples
    embeddings, colors = get_embeddings(autoencoder, testloader, device)
    # get the minimum and maximum xs and ys
    mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
    # get random samples from latent space using the max xs and max ys
    sample = np.random.uniform(mins, maxs, size=(18, 2))
    sample = torch.from_numpy(sample).to(torch.float32)
    autoencoder.decoder.eval()
    # put the latent space/embeddings into the decoder to create images
    with torch.no_grad():
        reconstructions = autoencoder.decoder(sample)
    # show images that the decoder created
    generate(reconstructions, sample)
    # get the embeddings of all the test samples
    embeddings, colors = get_embeddings(autoencoder, testloader, device)
    # plot latent space
    latent_space_show(embeddings, colors, sample)
    plt.show()
