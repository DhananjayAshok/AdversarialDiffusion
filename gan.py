import torch.nn as nn
import torch
from utils import Parameters, DummyModel
from numpy.random import choice
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


def conv_block(input_channels, output_channels):
    c1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=1, padding=2)
    r1 = nn.ReLU()
    c2 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=4, stride=1, padding=2)
    r2 = nn.ReLU()
    return c1, r1, c2, r2


def deconv_block(input_channels, output_channels):
    c1 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=1, padding=2)
    r1 = nn.ReLU()
    c2 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=4, stride=1, padding=2)
    r2 = nn.ReLU()
    return c1, r1, c2, r2


class Generator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        layers = []
        b1 = conv_block(input_channels=1, output_channels=5)
        layers.extend(b1)  # Shape here is bs, 5, 30, 30
        pooled = nn.MaxPool2d(kernel_size=2)  # bs, 5, 15, 15
        bn = nn.BatchNorm2d(5)
        layers.extend([pooled, bn])
        b2 = conv_block(input_channels=5, output_channels=10)
        pooled2 = nn.MaxPool2d(kernel_size=4)
        layers.extend(b2)
        layers.append(pooled2)
        # bs, 10, 4, 4
        unpool2 = nn.Upsample(scale_factor=4)
        layers.append(unpool2)
        b3 = deconv_block(input_channels=10, output_channels=5)
        layers.extend(b3)
        unpool3 = nn.Upsample(scale_factor=2)
        bn1 = nn.BatchNorm2d(5)
        layers.extend([unpool3, bn1])
        b4 = deconv_block(input_channels=5, output_channels=1)
        layers.extend(b4)
        final = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3)
        layers.append(final)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        layers = []
        b1 = conv_block(input_channels=1, output_channels=5)
        layers.extend(b1)  # Shape here is bs, 5, 30, 30
        pooled = nn.MaxPool2d(kernel_size=2)  # bs, 5, 15, 15
        bn = nn.BatchNorm2d(5)
        layers.extend([pooled, bn])
        b2 = conv_block(input_channels=5, output_channels=10)
        pooled2 = nn.MaxPool2d(kernel_size=4)
        layers.extend(b2)
        layers.append(pooled2)
        flat = nn.Flatten()
        dense = nn.Linear(10*4*4, 500)
        penultimate = nn.Linear(500, 10)
        drop = nn.Dropout()
        r = nn.ReLU()
        logits = nn.Linear(10, 2)
        layers.extend([flat, dense, penultimate, drop, r, logits])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_g(gen, opt, lr_sched, criterion, disc, mixture_dset, train_loader, val_loader, attack_set, max_iter=2):
    train_loss = []
    for iter in range(max_iter):
        for j, (idxs, batch_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
                rand_index = choice(range(len(mixture_dset.models)))
                s_model = mixture_dset.models[0][rand_index]

                # for now we batch all the examples from
                clean_imgs, true_labels = batch_data
                attacked_imgs = attack_set(s_model, clean_imgs, true_labels,
                                           preprocessing=mixture_dset.preprocessings[rand_index])

                clean_imgs = clean_imgs.to(Parameters.device)
                attacked_imgs = attacked_imgs.to(Parameters.device)

                attacked_imgs_hat = gen(clean_imgs)
                # train
                print(attacked_imgs_hat.shape, attacked_imgs.shape)
                loss_value = criterion(attacked_imgs_hat, clean_imgs)
                loss_value.backward()
                train_loss.append(loss_value.item())
                opt.step()
                print(train_loss[-1])
        lr_sched.step()
        print(f"Epoch Latest Loss: {train_loss[-10:-1]}")
    train_loss = np.array(train_loss)
    return train_loss


def train_d(disc, opt, lr_sched, criterion, gen, mixture_dset, train_loader, val_loader, attack_set, max_iter=2, device=Parameters.device):
    train_loss = []
    for iter in range(max_iter):
        for j, (idxs, batch_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
                rand_index = choice(range(len(mixture_dset.models)))
                s_model = mixture_dset.models[0][rand_index]

                # for now we batch all the examples from
                clean_imgs, true_labels = batch_data
                attacked_imgs = attack_set(s_model, clean_imgs, true_labels,
                                           preprocessing=mixture_dset.preprocessings[rand_index])

                clean_imgs = clean_imgs.to(Parameters.device)
                attacked_imgs = attacked_imgs.to(Parameters.device)

                attacked_imgs_hat = gen(clean_imgs)
                hat_labels = torch.ones(size=(attacked_imgs_hat.shape[0]))
                attacked_labels = torch.zeros_like(hat_labels)
                in_batch = torch.cat([attacked_imgs_hat, attacked_imgs], dim=0)
                out_labels = torch.cat([hat_labels, attacked_labels]).to(Parameters.device)
                pred = disc(in_batch)
                # train
                loss_value = criterion(pred, out_labels)
                loss_value.backward()
                train_loss.append(loss_value.item())
                opt.step()
        lr_sched.step()
        print(f"Epoch Latest Loss: {train_loss[-10:-1]}")
    train_loss = np.array(train_loss)
    return train_loss


def train(mixture_dset, attack_set, max_iter=5, max_internal_iter=2, save_name="", gan_warm_start=False, train_size=0.8):
    gen = Generator().to(Parameters.device)
    disc = Discriminator().to(Parameters.device)
    if gan_warm_start:
        state_dict = torch.load(f"models/gan/{save_name}_g.pth")
        gen.load_state_dict(state_dict)
    g_opt = torch.optim.SGD(gen.parameters(), lr=0.00001, momentum=0.3, weight_decay=0.1)
    g_lr_sched = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=0.95)
    g_criterion = nn.MSELoss()

    d_opt = torch.optim.SGD(disc.parameters(), lr=0.05, momentum=0.7, weight_decay=0.2)
    d_lr_sched = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.95)

    d_criterion = nn.CrossEntropyLoss()
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

    for i in tqdm(range(max_iter)):
        #d_curves = train_d(disc, d_opt, d_lr_sched, criterion, gen, mixture_dset, train_loader, val_loader, attack_set, max_iter=max_internal_iter)
        g_curves = train_g(gen, g_opt, g_lr_sched, g_criterion, disc, mixture_dset, train_loader, val_loader, attack_set, max_iter=max_internal_iter)
        print(g_curves[-1])
        torch.save(gen.state_dict(), f"models/gan/{save_name}_g.pth")
        #torch.save(disc.state_dict(), f"models/gan/{save_name}_d.pth")


def get_identity(diff_model_name, mixture_dset, attack_set, save_name=""):
    return DummyModel()
