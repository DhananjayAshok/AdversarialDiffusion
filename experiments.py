from attacks import ImageAttack, ATTACKS
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from utils import measure_attack_success, measure_attack_model_success, get_common, dict_to_namespace, measure_transfer_attack_success
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from numpy.random import choice
from utils import Parameters,safe_mkdir, DummyModel, count_parameters, measure_attack_model_epsilon_bound
import os
from tqdm import tqdm
from pathlib import Path
from torchvision.models import resnet50, resnet18, resnet34
import matplotlib.pyplot as plt
from vae import get_vae, load_vae
import pandas as pd
from attacks import AttackSet

from diffusion import get_diffusion, load_diffusion


def experiment_0(target_model_arch, attack, dataset_class):
    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_success = measure_attack_success(mixture_dset, attack_set, model)
    return attack_success


def transfer_experiment(target_model_arch1, target_model_arch2, attack, dataset_class):
    attack_set1, mixture_dset1 = get_common([target_model_arch1], [attack], [dataset_class], train=False)
    attack_set2, mixture_dset2 = get_common([target_model_arch2], [attack], [dataset_class], train=False)
    model1 = mixture_dset1.models[0][0]
    model2 = mixture_dset2.models[0][0]
    attack_success = measure_transfer_attack_success(model1, model2, mixture_dset1, attack)
    return attack_success


def experiment_1(diff_model_name, target_model_arch, attack, dataset_class,
                 experiment_name='single_model-single_attack-single_dataset', train=True):
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
        #attack_model = get_vae(diff_model_name, mixture_dset, attack_set)
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set, num_epochs = 30, save_name= experiment_name, batch_size= 512)
    else:
        ckpt = load_diffusion(diff_model_name, experiment_name)
        attack_model = ckpt[0]
        #attack_model = load_vae(diff_model_name, experiment_name)

    attack_model.sample_plot_image(IMG_SIZE = 64, filename = f'figures/{experiment_name}.png')

    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    clean_accuracy, robust_accuracy = measure_attack_success(mixture_dset, attack_set, model)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model, target_model=model)

    model_l2_norm, model_linf_norm = measure_attack_model_epsilon_bound(mixture_dset, attack_model)
    visualize_images(mixture_dset, attack_set, attack_model, filename = '/home/mmml/MMML/AdversarialDiffusion/models/' + model.__class__.__name__ + '/' +experiment_name + '/figures_with_labels/', no_limit=250, batch_size=32)
    return clean_accuracy, robust_accuracy, attack_model_success[1], model_l2_norm, model_linf_norm

import pdb
def experiment_2(diff_model_name, target_model_arch, attacks, dataset_class,
                 experiment_name = 'single_model-multiple_attack-single_dataset', train=True):
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
        #attack_model = get_vae(diff_model_name, mixture_dset, attack_set, experiment_name)
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set, save_name = experiment_name, num_epochs = 30)
    else:
        ckpt = load_diffusion(diff_model_name, experiment_name)
        attack_model = ckpt[0]
        #attack_model = load_vae(diff_model_name, experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model, target_model= model)

    clean_accuracy, robust_accuracy = measure_attack_success(mixture_dset, attack_set, target_model=model)

    model_l2_norm, model_linf_norm = measure_attack_model_epsilon_bound(mixture_dset, attack_model)
    visualize_images(mixture_dset, attack_set, attack_model, filename = '/home/mmml/MMML/AdversarialDiffusion/models/' + model.__class__.__name__ + '/' + experiment_name + '/figures_with_labels/', no_limit=250, batch_size=32)
    # attack_results = []
    # for attack in attacks: # what is the efficacy on every attack.
    #     attack_s = AttackSet([attack])
    #     attack_success = measure_attack_success(mixture_dset, attack_s, target_model=model)
    #     attack_results.append(attack_success)
    return clean_accuracy, robust_accuracy, attack_model_success[1], model_l2_norm, model_linf_norm

from utils import visualize_images
def experiment_3(diff_model_name, train_model_archs, attack, train_dataset_classes, test_model_archs,
                 test_dataset_classes, experiment_name, train=False):
    if len(test_dataset_classes) == 0:
        test_dataset_classes = train_dataset_classes
    if len(test_model_archs) == 0:
        test_model_archs = train_model_archs
    if train:
        attack_set, mixture_dset = get_common(train_model_archs, attack, train_dataset_classes, train=True)
        #attack_model = get_vae(diff_model_name, mixture_dset, attack_set)
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set, save_name=experiment_name, num_epochs = 30)
    else:
        ckpt = load_diffusion(diff_model_name, experiment_name)
        attack_model = ckpt[0]
        #attack_model = load_vae(diff_model_name, experiment_name)

    attack_set, mixture_dset = get_common(test_model_archs, attack, test_dataset_classes, train=False)
    attack_results = measure_attack_success(mixture_dset, attack_set)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model)

    model_l2_norm, model_linf_norm = measure_attack_model_epsilon_bound(mixture_dset, attack_model)
    visualize_images(mixture_dset, attack_set, attack_model, filename = '/home/mmml/MMML/AdversarialDiffusion/models/' + experiment_name + '/figures_with_labels/', no_limit=250, batch_size=32)
    return *attack_results, attack_model_success[1], model_l2_norm, model_linf_norm


def run_experiment0():
    target_model_archs = [(resnet18, "18"), (resnet34, "34"), (resnet50, "50")]
    attacks = [ATTACKS['pgd']]#, ATTACKS["fgsm"]]
    epsilons = [0.01*i for i in range(2)]
    dataset_classes = [MNIST, KMNIST]
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
    attacks = [ATTACKS['pgd'], ATTACKS["fgsm"]]
    epsilons = [0.01*i for i in range(20)]
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


def run_experiment1():
    target_model_archs = [(resnet18, "18")]
    attacks = [ATTACKS["fgsm"], ATTACKS['pgd']]
    dataset_classes = [KMNIST, MNIST]
    columns = ["dataset", "attack", "target_model", "clean_accuracy", "robust_accuracy", "model_robust_accuracy", 'l2_norm', 'linf_norm']
    data = []
    for dataset_class in dataset_classes:
        for attack in attacks:
            for target_model_arch in target_model_archs:
                name = f"ResNet{target_model_arch[1]}"
                print('Experiment 1: ', dataset_class, attack, target_model_arch)
                clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_1('simple_diffnet',
                                                                                                          target_model_arch, attack,
                                                                                                          dataset_class,
                                                                                                          experiment_name= f"{name}_{dataset_class.__name__}_{attack.__name__}",
                                                                                                          train = False)
                data.append([dataset_class.__name__, attack.__name__, name, clean_accuracy, robust_accuracy,
                             model_robust_accuracy, l2_norm, linf_norm])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_1.csv", index=False)
    return df


def run_experiment2():
    target_model_archs = [(resnet18, "18")]
    attacks = [ATTACKS['pgd'], ATTACKS["fgsm"]]
    dataset_classes = [MNIST]#, KMNIST]
    columns = ["dataset", "target_model", "clean_accuracy", "robust_accuracy", "model_robust_accuracy", 'l2_norm', 'linf_norm']
    data = []
    for dataset_class in dataset_classes:
        for target_model_arch in target_model_archs:
            name = f"ResNet{target_model_arch[1]}"
            print('Experiment 2: ', dataset_class, attacks, target_model_arch)
            clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_2("simple_diffnet",
                                                                                  target_model_arch, attacks,
                                                                                  dataset_class, f"{name}_{dataset_class.__name__}",
                                                                                  train = False)
            data.append([dataset_class.__name__, name, clean_accuracy, robust_accuracy,
                         model_robust_accuracy, l2_norm, linf_norm])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_2.csv", index=False)
    return df


def run_experiment3():
    # dataset mixture train and test
    target_model_archs = [(resnet18, "18")]
    train_dataset_classes = [MNIST, KMNIST]
    test_dataset_classes = [FashionMNIST]
    attacks = [ATTACKS['pgd']]
    columns = ["config", "clean_accuracy", "robust_accuracy", "model_robust_accuracy", 'l2_norm', 'linf_norm']
    data = []
    print('Experiment 3: ')
    clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_3(diff_model_name="simple_diffnet",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks,
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=train_dataset_classes,
                                                                          experiment_name='single_model-single_attack-multiple_dataset',
                                                                          train = False)
    data.append(["Train", clean_accuracy, robust_accuracy,
                 model_robust_accuracy, l2_norm, linf_norm])

    clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_3(diff_model_name="simple_diffnet",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks,
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=test_dataset_classes,
                                                                          experiment_name="single_model-single_attack-multiple_dataset",
                                                                          train=False)
    data.append(["Test", clean_accuracy, robust_accuracy,
                 model_robust_accuracy, l2_norm, linf_norm])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_3.csv", index=False)
    return df


def run_experiment4():
    # model mixture train and test
    target_model_archs = [(resnet18, "18"), (resnet34, "34")]
    test_model_archs = [(resnet50, "50")]
    train_dataset_classes = [MNIST]
    attacks = [ATTACKS['pgd']]
    columns = ["config", "clean_accuracy", "robust_accuracy", "model_robust_accuracy", 'l2_norm', 'linf_norm']
    data = []
    print('Experiment 4: ')
    clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_3(diff_model_name=f"simple_diffnet",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks,
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          experiment_name="multiple_model-single_attack-single_dataset",
                                                                          test_dataset_classes=train_dataset_classes,
                                                                          train = False)
    data.append(["Train", clean_accuracy, robust_accuracy,
                 model_robust_accuracy, l2_norm, linf_norm])

    clean_accuracy, robust_accuracy, model_robust_accuracy, l2_norm, linf_norm = experiment_3(diff_model_name="simple_diffnet",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks,
                                                                          test_model_archs=test_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=train_dataset_classes,
                                                                          experiment_name="multiple_model-single_attack-single_dataset",
                                                                          train=False)
    data.append(["Test", clean_accuracy, robust_accuracy,
                 model_robust_accuracy, l2_norm, linf_norm])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_4.csv", index=False)
    return df


if __name__ == "__main__":
    run_experiment1()
    run_experiment2()
    run_experiment3()
    run_experiment4()
    # attack_success = experiment_0(target_model_arch = (resnet18, "18"), attack = ATTACKS['pgd'], dataset_class = KMNIST)
    # print(attack_success)