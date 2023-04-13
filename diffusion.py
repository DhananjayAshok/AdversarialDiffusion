from utils import measure_attack_success, measure_attack_model_success, get_common, dict_to_namespace, measure_transfer_attack_success
from DiffPure import DiffPure
from denoising_diff import DDPM
import yaml
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from numpy.random import choice
from utils import Parameters,safe_mkdir, count_parameters
import os
from tqdm import tqdm
from pathlib import Path
from torchvision.models import resnet50, resnet18, resnet34
import matplotlib.pyplot as plt
import pdb
import time
from simple_diffnet import SimpleDiff #SimpleUnet, forward_diffusion_sample
from collections import defaultdict

def plotting(X, advs, model_attacked, filename):
    indices = torch.randint(0, len(X), size = (10,))
    X = X[indices]
    fig, axes = plt.subplots(nrows = 10, ncols = 3, figsize = (12, 3 * 10))
    for i in range(10):
        ax = axes[i]
        # display the image X[i]
        ax[0].imshow(X[i][0], cmap = 'gray')
        ax[0].set_title('Clean image')#, fontsize = 18)
        ax[0].axis('off')

        # display the adversarial example advs[i]
        ax[1].imshow(advs[i][0], cmap = 'gray')
        ax[1].set_title(f'Adversarial image')#, fontsize = 18)
        ax[1].axis('off')

        # display the attacked image
        ax[2].imshow(model_attacked[i][0], cmap = 'gray')
        ax[2].set_title(f'Model attacked')#, fontsize = 18)
        ax[2].axis('off')

    fig.subplots_adjust(wspace=0.5, hspace = 0.5)
    fig.savefig(filename, bbox_inches='tight')


def get_diffusion(diff_model_name, mixture_dset, attack_set, num_epochs=15, batch_size=256, lr=0.0001, train_size=0.9,
          early_stopping=5,
          save_name=""):

    if diff_model_name in ['guided_diffusion']:
        config_file = f'DiffPure/configs/{diff_model_name}_config.yml'

        with open(config_file, 'r') as stream:
            config_dict = yaml.safe_load(stream)

        # Convert the dictionary to a namespace object
        config_namespace = dict_to_namespace(config_dict)
        args = config_namespace.args
        config = config_namespace.config

        # load diffpure model.
        attack_model = DiffPure(args, config)
        # pdb.set_trace()

    elif diff_model_name == 'ddpm':
        attack_model = DDPM()
        print(f'Loaded model {diff_model_name} | num params: {count_parameters(attack_model)//10e6}M')

    elif diff_model_name == 'simple_diffnet':
        attack_model = SimpleDiff()

    # attack_model = DummyModel()

    # from torchsummary import summary
    # print(summary(attack_model, input_size=(3, 64, 64)))

    # sanity checks.
    # image = torch.from_numpy(np.random.randint(0, 255, size=(5, 3, 64, 64)).astype(np.uint8))
    # out = attack_model(image)
    # print(out.shape) # this should work for you. ensure that, if not reach out.
    # also the shape should be (1, 64, 64) but i couldnt check that.
    # if not we need to add a conv layer to make it that way.

    whole_train_dataset = mixture_dset
    train_size_int = int(len(whole_train_dataset) * train_size)
    rand_indices = np.arange(len(whole_train_dataset))
    np.random.shuffle(rand_indices)
    train_indices = rand_indices[:train_size_int]
    val_indices = rand_indices[train_size_int:]
    train_dataset = Subset(whole_train_dataset, train_indices)
    val_dataset = Subset(whole_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True)
    attack_model = attack_model.to(Parameters.device)
    attack_model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    save_suffix = ""
    if save_name != "":
        save_suffix = f"_{save_name}"
    checkpoint_path = os.path.join(Parameters.model_path, diff_model_name+save_suffix)
    final_path = os.path.join(Parameters.model_path, diff_model_name+save_suffix,
                              "final_chkpt.pt")
    figures_dir = os.path.join(Parameters.model_path, diff_model_name+save_suffix, 'figures')
    safe_mkdir(checkpoint_path)
    safe_mkdir(figures_dir)

    n_total_step = len(train_loader)
    logs = defaultdict(dict)

    best_val_loss = np.inf
    epochs_without_improvement = 0
    print(f"Starting Training for Mixture class on {attack_model.__class__.__name__+save_suffix}")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        attack_model.train()
        # iterate over batch.
        for i, (idxs, batch_data) in tqdm(enumerate(train_loader), total = n_total_step):
            attacked_images = torch.zeros_like(batch_data[0]).to(Parameters.device)

            for idx in set(idxs.tolist()):
                s_model = choice(mixture_dset.models[idx])
                idx_mask = (idxs == idx)
                X_slice = batch_data[0][idx_mask].to(Parameters.device)
                y_slice = batch_data[1][idx_mask].to(Parameters.device)
                attacked_images[idx_mask] = attack_set(s_model, X_slice, y_slice,
                                                    preprocessing=mixture_dset.preprocessings[idx])

            clean_images = batch_data[0].to(Parameters.device)
            # clean_images = 2 * clean_images - 1

            predicted_noise = attack_model(clean_images)
            # target_noise = attacked_images - clean_images

            # train
            loss_value = criterion(predicted_noise, attacked_images) # target_noise)
            loss_value.backward()
            optimizer.step()
            print('loss: ', loss_value.item())

            logs[epoch * num_epochs + i]['train_loss'] = loss_value.item()

            # success of this attack.
            if (i + 1) % 50 == 0:
                # preds
                filename = os.path.join(figures_dir, f'epoch{epoch+1}_step{i+1}.png')
                clean_accuracy, robust_accuracy = measure_attack_model_success(mixture_dset, attack_model)
                plotting(clean_images.detach().cpu().numpy(),
                         advs=attacked_images.detach().cpu().numpy(),
                         model_attacked=(predicted_noise + clean_images).detach().cpu().numpy(),
                         filename = filename)
                logs[epoch * num_epochs + i]['train_clean_acc'] = clean_accuracy
                logs[epoch * num_epochs + i]['train_robust_acc'] = robust_accuracy
                print(f'epoch {epoch + 1}/{num_epochs}, step: {i + 1}/{n_total_step}: loss = {loss_value.item():.5f}, '
                      f'clean accuracy = {100 * clean_accuracy:.2f}, '
                      f'robust accuracy = {100 * robust_accuracy:.2f}'
                )
                torch.save({'attack_model': attack_model.state_dict(), 'logs': logs}, os.path.join(checkpoint_path, f"chkpt_{epoch}.pth"))
            del clean_images
            del predicted_noise
            del attacked_images

        scheduler.step()

        val_loss, val_clean_acc, val_robust_acc = _val(attack_model, val_loader, mixture_dset, attack_set, criterion)
        logs[epoch * num_epochs + i]['val_loss'] = loss_value.item()
        logs[epoch * num_epochs + i]['val_clean_acc'] = val_clean_acc
        logs[epoch * num_epochs + i]['val_robust_acc'] = val_robust_acc
        print(f'epoch {epoch + 1}/{num_epochs}: val loss = {val_loss:.5f}, '
                      f'val clean accuracy = {100 * val_clean_acc:.2f}, '
                      f'val robust accuracy = {100 * val_robust_acc:.2f}'
                )
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
        if early_stopping is not None:
            if epochs_without_improvement >= early_stopping:
                print(f"Early Stopping...")
                break

    torch.save({'attack_model': attack_model.state_dict(), 'logs': logs}, final_path)
    # test_loss = _val(attack_model, test_loader, criterion)
    # return test_loss
    return attack_model

def _val(model, test_loader, mixture_dset, attack_set, criterion):
    val_losses = []

    print('calculating val loss.')
    model.eval()
    for idxs, batch_data in test_loader:
        X, y = batch_data
        X = X.to(Parameters.device)
        y = y.to(Parameters.device)
        attacked_images = torch.zeros_like(X).to(Parameters.device)
        target_noise = attacked_images - X

        for idx in set(idxs.tolist()):
            s_model = choice(mixture_dset.models[idx])
            idx_mask = (idxs == idx)
            X_slice = X[idx_mask]
            y_slice = y[idx_mask]

            attacked_images[idx_mask] = attack_set(s_model, X_slice, y_slice,
                                                preprocessing=mixture_dset.preprocessings[idx])
        with torch.no_grad():
            predicted_noise = model(X)

        loss_value = criterion(predicted_noise, attacked_images)#target_noise)
        val_losses.append(loss_value.item())

    print('finding val attack model success.')
    clean_acc, robust_acc = measure_attack_model_success(test_loader.dataset.dataset, model)
    return np.mean(val_losses), clean_acc, robust_acc


def load_diffusion(diff_model_name, save_name):
    save_suffix = ""
    if save_name != "":
        save_suffix = f"_{save_name}"

    final_path = os.path.join(Parameters.model_path, diff_model_name + save_suffix,
                              "final_chkpt.pt")
    print(final_path)
    assert Path(final_path).exists()

    ckpt = torch.load(final_path)
    state_dict = ckpt['attack_model']

    if diff_model_name in ['guided_diffusion']:
        config_file = f'DiffPure/{diff_model_name}_config.yml'
        with open(config_file, 'r') as stream:
            config_dict = yaml.safe_load(stream)
        config_namespace = dict_to_namespace(config_dict)
        args = config_namespace.args
        config = config_namespace.config

        # create model instance.
        attack_model = DiffPure(args, config)

    elif diff_model_name == 'ddpm':
        attack_model = DDPM()
        print(f'Loaded model {diff_model_name} | num params: {count_parameters(attack_model)//10e6}M')

    elif diff_model_name == 'simple_diffnet':
        attack_model = SimpleDiff()

    # load in the weights
    attack_model.load_state_dict(state_dict)
    attack_model = attack_model.to(Parameters.device)

    return attack_model