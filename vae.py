import torch
import torch.nn as nn
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader
from collections import defaultdict

import utils


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x.reshape(-1, 1, 28, 28), means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x.reshape(-1, 1, 28, 28)


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class VAEHolder(nn.Module):
    def __init__(self, vae):
        nn.Module.__init__(self)
        self.vae = vae

    def forward(self, x, y):
        recon, mean, var, z = self.vae(x, y)
        return recon.reshape(-1, 1, 28, 28) + x

    def load(self, name="ResNet18_MNIST_PGDL2"):
        self.vae.load_state_dict(torch.load(f"models/vae_{name}/final_model.pth"))


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)


def load_vae(model_name, exp_name):
    vae = VAE(
        encoder_layer_sizes=[784, 256],
        latent_size=2,
        decoder_layer_sizes=[256, 784],
        conditional=True,
        num_labels=10).to(utils.Parameters.device)
    vae = VAEHolder(vae)
    vae.load(model_name)
    return vae


def get_vae(model_name, mixture_dset, attack_set, epochs=3, latent_size=2, train_size=0.8):
    whole_train_dataset = mixture_dset
    train_size_int = int(len(whole_train_dataset) * train_size)
    rand_indices = np.arange(len(whole_train_dataset))
    np.random.shuffle(rand_indices)
    train_indices = rand_indices[:train_size_int]
    val_indices = rand_indices[train_size_int:]
    train_dataset = Subset(whole_train_dataset, train_indices)
    val_dataset = Subset(whole_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=True)
    plot_path = f"figures/vae_{model_name}/"
    model_path = f"models/vae_{model_name}/"
    utils.safe_mkdir(path=plot_path)
    utils.safe_mkdir(path=model_path)
    device = utils.Parameters.device
    vae = VAE(
        encoder_layer_sizes=[784, 256],
        latent_size=latent_size,
        decoder_layer_sizes=[256, 784],
        conditional=True,
        num_labels=10).to(device)
    if os.path.exists(model_path+"final_model.pth"):
        vae = VAEHolder(vae)
        vae.load(name=model_name)
        return vae

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    logs = defaultdict(list)

    for epoch in range(epochs):
        for iteration, (info) in tqdm(enumerate(train_loader), total=len(train_loader)):
            idx, (X, Y) = info
            X, Y = X.to(device), Y.to(device)
            attacked_imgs = None
            recon_x = None
            x = None
            for index in range(len(mixture_dset.models)):
                s_model = np.random.choice(mixture_dset.models[index])
                idx_mask = (index == idx)
                x = X[idx_mask]
                y = Y[idx_mask]
                attacked_imgs = attack_set(s_model, x, y, preprocessing=mixture_dset.preprocessings[index])
                attacked_imgs = attacked_imgs.to(device)

                recon_x, mean, log_var, z = vae(x, y)
                loss = loss_fn(recon_x, attacked_imgs-x, mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % 500 == 0 or iteration == len(train_loader) - 1:
                print(loss.item())
                utils.save_img(x[0], save_path=plot_path+f"{epoch}_{iteration}_x.png", title="x")
                utils.save_img(attacked_imgs[0], save_path=plot_path+f"{epoch}_{iteration}_attacked.png", title="Adv")
                utils.save_img(attacked_imgs[0] - x[0], save_path=plot_path+f"{epoch}_{iteration}_target.png",
                               title="Target")
                utils.save_img(recon_x[0], save_path=plot_path+f"{epoch}_{iteration}_output.png",
                               title="Output")

                c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                z = torch.randn([c.size(0), latent_size]).to(device)
                x = vae.inference(z, c=c)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p + 1)
                    plt.text(
                        0, 0, "c={:d}".format(c[p].item()), color='black',
                        backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).cpu().data.numpy())
                    plt.axis('off')


                plt.savefig(
                    os.path.join(plot_path,
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
                state_dict = vae.state_dict()
                torch.save(state_dict, f"{model_path}/chkpt_{epoch}_{iteration}.pth")
        vae.eval()
        avg_loss = 0
        for iteration, (info) in tqdm(enumerate(val_loader), total=len(val_loader)):
            idx, (X, Y) = info
            X, Y = X.to(device), Y.to(device)

            for index in range(len(mixture_dset.models)):
                s_model = np.random.choice(mixture_dset.models[index])
                idx_mask = (index == idx)
                x = X[idx_mask]
                y = Y[idx_mask]
                attacked_imgs = attack_set(s_model, x, y, preprocessing=mixture_dset.preprocessings[index])
                attacked_imgs = attacked_imgs.to(device)

                recon_x, mean, log_var, z = vae(x, y)
                loss = loss_fn(recon_x, attacked_imgs, mean, log_var)
                avg_loss += loss.item()
        print(f"Validation Loss Average: {avg_loss/len(val_loader)}")
        vae.train()
    torch.save(state_dict, f"{model_path}/final_model.pth")
    return VAEHolder(vae)
