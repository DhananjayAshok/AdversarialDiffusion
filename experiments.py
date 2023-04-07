import pdb
from attacks import ImageAttack, ATTACKS
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from utils import measure_attack_success, measure_attack_model_success, get_common, dict_to_namespace, measure_transfer_attack_success
from DiffPure import DiffPure
from DDPM import DDPM
import yaml
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from numpy.random import choice
from utils import Parameters,safe_mkdir, DummyModel, count_parameters
import os
from tqdm import tqdm
from pathlib import Path
from torchvision.models import resnet50, resnet18, resnet34
import matplotlib.pyplot as plt
from gan import train as get_gan

def experiment_0(target_model_arch, attack, dataset_class):
    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_success = measure_attack_success(model, mixture_dset, attack)
    return attack_success


def transfer_experiment(target_model_arch1, target_model_arch2, attack, dataset_class):
    attack_set1, mixture_dset1 = get_common([target_model_arch1], [attack], [dataset_class], train=False)
    attack_set2, mixture_dset2 = get_common([target_model_arch2], [attack], [dataset_class], train=False)
    model1 = mixture_dset1.models[0][0]
    model2 = mixture_dset2.models[0][0]
    attack_success = measure_transfer_attack_success(model1, model2, mixture_dset1, attack)
    return attack_success


def experiment_1(diff_model_name, target_model_arch, attack, dataset_class,
                 experiment_name = 'single_model-single_attack-single_dataset', train=True):
    '''
    Single target model, single attack on single dataset.
    params:
        target_model_arch: one target classifier.
        attack: corresponding i th attack will be used to generate adversarial examples.
        dataset_class: corresponding i th dataset class.
        experiment_name: auto generate as combo of target_model_arch-attack-dataset_class-single
        train: whether to train the model or not.
    '''
    if train:
        attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=train)
        attack_model = attack_model = get_diffusion(diff_model_name, mixture_dset,
                                                    attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(diff_model_name, experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_success = measure_attack_success(model, mixture_dset, attack)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model, model=model)
    return attack_success, attack_model_success


def experiment_2(diff_model_name, target_model_arch, attacks, dataset_class,
                 experiment_name = 'single_model-multiple_attack-single_dataset', train=False):
    '''
    single target model, multiple attack on single dataset.

    [target_model_arch], attacks, [dataset_class]
    params:
        target_model_arch: one target classifier.
        attack: corresponding i th attack will be used to generate adversarial examples.
        dataset_class: corresponding i th dataset class.
        experiment_name: auto generate as combo of target_model_arch-attack-dataset_class-single
        train: whether to train the model or not.
    '''
    if train:
        attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class], train=True)
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(diff_model_name, experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model)
    attack_results = []
    for attack in attacks: # what is the efficacy on every attack.
        attack_success = measure_attack_success(mixture_dset, attack, model=model)
        attack_results.append(attack_success)
    return attack_results, attack_model_success


def experiment_3(train_model_archs, attack, train_dataset_classes, test_model_archs,
                 test_dataset_classes, experiment_name, train=False):
    if len(test_dataset_classes) == 0:
        test_dataset_classes = train_dataset_classes
    if len(test_model_archs) == 0:
        test_model_archs = train_model_archs
    if train:
        attack_set, mixture_dset = get_common(train_model_archs, attack, train_dataset_classes, train=True)
        attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(experiment_name)
    attack_set, mixture_dset = get_common(test_model_archs, attack, test_dataset_classes, train=True)
    attack_results = measure_attack_success(mixture_dset, attack)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model)
    return attack_results, attack_model_success


def run_experiment1():
    target_model_arch = resnet50
    attack = ImageAttack(ATTACKS['pgd_l2'])
    dataset_class = MNIST
    experiment_name = "first"
    train = True
    a, b = experiment_1('ddpm', (target_model_arch, '50'), attack, dataset_class,
                        experiment_name, train=train)
    print(a)
    print(b)
    return a, b


def run_experiment0():
    target_model_archs = [(resnet18, "18"), (resnet34, "34"), (resnet50, "50")]
    attacks = [ATTACKS['pgd_l2'], ATTACKS["fgsm"]]
    epsilons = [0.5*i for i in range(20)]
    dataset_classes = [KMNIST]
    for dataset_class in dataset_classes:
        for attack in attacks:
            for target_model_arch in target_model_archs:
                name = f"ResNet{target_model_arch[1]}"
                means = []
                stds = []
                ep = []
                for eps in epsilons:
                    params = {"eps": eps}
                    attack_fix = ImageAttack(attack, params=params)
                    a, b = experiment_0(target_model_arch, attack_fix, dataset_class)
                    if len(means) == 0:
                        means.append(a.mean())
                        stds.append(a.std())
                        ep.append(0)
                    means.append(b.mean())
                    stds.append(b.std())
                    ep.append(eps)
                plt.errorbar(ep, means, stds, linestyle='None', marker='^')
                plt.title(f"{attack.__name__} on {dataset_class.__name__}  {name}")
                plt.xlabel(f"Epsilon")
                plt.ylabel(f"Accuracy")
                #plt.show()
                print(f"Saving to figures/{attack.__name__} on {dataset_class.__name__}  {name}.png")
                plt.savefig(f"figures/{dataset_class.__name__}{attack.__name__}_{name}.png")
                plt.clf()
    return


def run_transfer_experiment():
    target_model_archs = [(resnet18, "18")]
    attacks = [ATTACKS['pgd_l2'], ATTACKS["fgsm"]]
    epsilons = [0.5*i for i in range(20)]
    dataset_classes = [MNIST, KMNIST]
    for dataset_class in dataset_classes:
        for attack in attacks:
            for target_model_arch in target_model_archs:
                means = []
                stds = []
                ep = []
                for eps in epsilons:
                    print(f"{ep} Done")
                    params = {"eps": eps}
                    attack_fix = ImageAttack(attack, params=params)
                    a, b = transfer_experiment(target_model_arch, (resnet34, "34"), attack_fix, dataset_class)
                    if len(means) == 0:
                        means.append(a.mean())
                        stds.append(a.std())
                        ep.append(0)
                    means.append(b.mean())
                    stds.append(b.std())
                    ep.append(eps)
                plt.errorbar(ep, means, stds, linestyle='None', marker='^')
                title = f"Transfer Attack {attack.__name__} on {dataset_class.__name__}"
                plt.title(title)
                plt.xlabel(f"Epsilon")
                plt.ylabel(f"Accuracy")
                print(f"Saving to figures/{title}.png")
                plt.savefig(f"figures/TA_{attack.__name__}_{dataset_class.__name__}.png")
                plt.clf()


if __name__ == "__main__":
    target_model_arch, attack, dataset_class = (resnet18, "18"), ATTACKS['fgsm'], MNIST
    experiment_1(None, target_model_arch, attack, dataset_class)
