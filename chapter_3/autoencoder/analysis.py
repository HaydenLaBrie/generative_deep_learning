import torch
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import Autoencoder, Encoder, Decoder, testloader


def get_embeddings(model, testloader, device):
    embeddings = []
    colors = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model.encoder(x_batch)
            pred = pred.tolist()
            y_batch = y_batch.tolist()
            # batch_data = list(zip(pred, y_batch))
            # embeddings.extend(batch_data)
            embeddings.extend(pred)
            colors.extend(y_batch)
            if len(embeddings) >= 5000:
                break
    embeddings = np.array(embeddings)
    colors = np.array(colors)
    return embeddings, colors


def plot_color_latent_space(embeddings, colors):
    plt.figure(figsize=(8, 8))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap="rainbow",
        c=colors,
        alpha=0.8,
        s=3
    )
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    device = "cpu"
    # load back the model
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = Autoencoder(encoder, decoder).to(device)
    state_dict = torch.load('autoencoder.pth')
    autoencoder.load_state_dict(state_dict)
    # get the embeddings of all the test samples
    embeddings, colors = get_embeddings(autoencoder, testloader, device)
    # plot latent space
    plot_color_latent_space(embeddings, colors)
    a = 1
