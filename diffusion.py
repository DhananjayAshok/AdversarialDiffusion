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

def get_diffusion(diff_model_name, mixture_dset, attack_set, num_epochs=5, batch_size=32, lr=0.001, train_size=0.9,
          early_stopping=1,
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

    elif diff_model_name == 'ddpm':
        attack_model = DDPM()
        print(f'Loaded model {diff_model_name} | num params: {count_parameters(attack_model)//10e6}M')

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
                              batch_size=4,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=True)

    attack_model = attack_model.to(Parameters.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    save_suffix = ""
    if save_name != "":
        save_suffix = f"_{save_name}"
    checkpoint_path = os.path.join(Parameters.model_path, diff_model_name+save_suffix)
    final_path = os.path.join(Parameters.model_path, diff_model_name+save_suffix,
                              "final_chkpt.pt")
    safe_mkdir(checkpoint_path)

    n_total_step = len(train_loader)

    best_val_loss = np.inf
    epochs_without_improvement = 0
    print(f"Starting Training for Mixture class on {attack_model.__class__.__name__+save_suffix}")
    for epoch in range(num_epochs):
        # iterate over batch.
        for i, (idxs, batch_data) in tqdm(enumerate(train_loader), total = n_total_step):
            rand_index = choice(range(len(mixture_dset.models)))
            # pdb.set_trace()
            s_model = mixture_dset.models[0][rand_index]

            # for now we batch all the examples from
            clean_imgs, true_labels = batch_data
            attacked_imgs = attack_set(s_model, clean_imgs, true_labels,
                                      preprocessing=mixture_dset.preprocessings[rand_index])

            clean_imgs = clean_imgs.to(Parameters.device)
            attacked_imgs = attacked_imgs.to(Parameters.device)

            # attacked_imgs = []
            # for j, idx in enumerate(idxs):
            #     clean_img, true_label = batch_data[0][j:j + 1], batch_data[1][j:j + 1]
            #     # pdb.set_trace()
            #     s_model = choice(mixture_dset.models[idx])
            #     attacked_img = attack_set(s_model, clean_img, true_label,
            #                               preprocessing=mixture_dset.preprocessings[idx])
            #     attacked_imgs.append(attacked_img)
            # attacked_imgs = torch.concat(attacked_imgs, dim=0).to(device)
            # clean_imgs = batch_data[0].to(device)

            # print(clean_imgs.device)
            # print(attack_model.runner.model.time_embed[0])
            # print(attack_model.runner.model.time_embed[0].weight.device)

            attacked_imgs_hat = attack_model(clean_imgs)
            # train
            loss_value = criterion(attacked_imgs_hat, attacked_imgs)
            loss_value.backward()
            optimizer.step()

            # success of this attack.
            clean_accuracy, robust_accuracy = measure_attack_model_success(mixture_dset, attack_model)
            if (i + 1) % 250 == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, '
                      f'clean accuracy = {100 * clean_accuracy:.2f}'
                      f'robust accuracy = {100 * robust_accuracy:.2f}'
                      )
            if (i + 1) % 500 == 0:
                torch.save(attack_model.state_dict(), os.path.join(checkpoint_path, f"chkpt_{epoch}.pth"))
            del clean_imgs
            del attacked_imgs_hat
            del attacked_imgs

        scheduler.step()

        val_loss = _val(attack_model, val_loader, mixture_dset, attack_set, criterion)
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
        if early_stopping is not None:
            if epochs_without_improvement >= early_stopping:
                print(f"Early Stopping...")
                break

    torch.save(attack_model.state_dict(), final_path)
    # test_loss = _val(attack_model, test_loader, criterion)
    # return test_loss

def _val(model, test_loader, mixture_dset, attack_set, criterion):
    with torch.no_grad():
        val_losses = []
        clean_accs = []
        robust_accs = []
        for idxs, batch_data in test_loader:
            rand_index = choice(range(len(mixture_dset.models)))

            s_model = mixture_dset.models[0][rand_index]

            # for now we batch all the examples from
            clean_imgs, true_labels = batch_data
            attacked_imgs = attack_set(s_model, clean_imgs, true_labels,
                                       preprocessing=mixture_dset.preprocessings[rand_index])
            attacked_imgs = attacked_imgs.to(Parameters.device)

            # attacked_imgs = []
            # for idx in range(len(idxs)):
            #     clean_img, true_label = batch_data[0][idx:idx + 1], batch_data[1][idx:idx + 1]
            #     s_model = choice(mixture_dset.models[idx])
            #     attacked_img = attack_set(s_model, clean_img, true_label,
            #                                      preprocessing=mixture_dset.preprocessings[idx])
            #     attacked_imgs.append(attacked_img)
            # attacked_imgs = torch.stack(attacked_imgs, dim=0).to(device)

            clean_imgs = batch_data[0].to(Parameters.device)
            attacked_imgs_predicted = model(clean_imgs)
            loss_value = criterion(attacked_imgs_predicted, attacked_imgs)
            clean_acc, robust_acc = measure_attack_model_success(test_loader, mixture_dset, model)
            clean_accs.append(clean_acc)
            robust_accs.append(robust_acc)
            val_losses.append(loss_value)
        print(f'Clean accuracy: {(sum(clean_accs) / len(clean_accs)) * 100} | Robust accuracy: {(sum(robust_accs) / len(robust_accs)) * 100} | % Loss: {loss_value}')
        return sum(val_losses) / len(val_losses)


def load_diffusion(diff_model_name, save_name):
    save_suffix = ""
    if save_name != "":
        save_suffix = f"_{save_name}"

    checkpoint_path = os.path.join(Parameters.model_path, diff_model_name + save_suffix)
    final_path = os.path.join(Parameters.model_path, diff_model_name + save_suffix,
                              "final_chkpt.pt")

    assert Path(final_path).exists()

    state_dict = torch.load(final_path)

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

    # load in the weights
    attack_model.load_state_dict(state_dict)

    return attack_model