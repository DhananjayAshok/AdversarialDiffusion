from attacks import ImageAttack, ATTACKS
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from utils import measure_attack_success, measure_attack_model_success, get_common, dict_to_namespace, measure_transfer_attack_success
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
from gan import get_identity
import pandas as pd

from diffusion import get_diffusion, load_diffusion


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
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set)
    else:
        attack_model = load_diffusion(diff_model_name, experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    clean_accuracy, robust_accuracy = measure_attack_success(model, mixture_dset, attack_set)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model, target_model=model)
    print(attack_model_success)
    return clean_accuracy, robust_accuracy, attack_model_success[1]


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
        attack_success = measure_attack_success(mixture_dset, attack, target_model=model)
        attack_results.append(attack_success)
    return attack_results, attack_model_success


def experiment_3(diff_model_name, train_model_archs, attack, train_dataset_classes, test_model_archs,
                 test_dataset_classes, experiment_name, train=False):
    if len(test_dataset_classes) == 0:
        test_dataset_classes = train_dataset_classes
    if len(test_model_archs) == 0:
        test_model_archs = train_model_archs
    if train:
        attack_set, mixture_dset = get_common(train_model_archs, attack, train_dataset_classes, train=True)
        attack_model = get_diffusion(diff_model_name, mixture_dset, attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(diff_model_name, experiment_name)
    attack_set, mixture_dset = get_common(test_model_archs, attack, test_dataset_classes, train=True)
    attack_results = measure_attack_success(mixture_dset, attack)
    attack_model_success = measure_attack_model_success(mixture_dset, attack_model)
    return attack_results, attack_model_success


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


def run_experiment1():
    target_model_archs = [(resnet18, "18")]#, (resnet34, "34"), (resnet50, "50")]
    attacks = [ATTACKS['pgd_l2'], ATTACKS["fgsm"]]
    dataset_classes = [MNIST, KMNIST]#, FashionMNIST]
    columns = ["dataset", "attack", "target_model", "clean_accuracy", "robust_accuracy", "model_robust_accuracy"]
    data = []
    for dataset_class in dataset_classes:
        for attack in attacks:
            for target_model_arch in target_model_archs:
                name = f"ResNet{target_model_arch[1]}"
                clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_1('simple_diffnet', target_model_arch, attack, dataset_class, f"{name}_{dataset_class.__name__}_{attack.__name__}")
                data.append([dataset_class.__name__, attack.__name__, name, clean_accuracy, robust_accuracy,
                             model_robust_accuracy])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_1.csv", index=False)
    return df


def run_experiment2():
    target_model_archs = [(resnet18, "18")]#, (resnet34, "34"), (resnet50, "50")]
    attacks = [ATTACKS['pgd_l2'], ATTACKS["fgsm"]]
    dataset_classes = [MNIST, KMNIST]#, FashionMNIST]
    columns = ["dataset", "target_model", "clean_accuracy", "robust_accuracy", "model_robust_accuracy"]
    data = []
    for dataset_class in dataset_classes:
        for target_model_arch in target_model_archs:
            name = f"ResNet{target_model_arch[1]}"
            clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_2(f"{name}_{dataset_class.__name__}",
                                                                                  target_model_arch, attacks,
                                                                                  dataset_class)
            data.append([dataset_class.__name__, name, clean_accuracy, robust_accuracy,
                         model_robust_accuracy])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_2.csv", index=False)
    return df



def run_experiment3():
    # dataset mixture train and test
    target_model_archs = [(resnet18, "18")]
    train_dataset_classes = [MNIST, KMNIST]
    test_dataset_classes = [FashionMNIST]
    attacks = [ATTACKS['pgd_l2']]
    columns = ["config", "clean_accuracy", "robust_accuracy", "model_robust_accuracy"]
    data = []
    clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_3(diff_model_name="e3_model",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks[0],
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=train_dataset_classes,
                                                                          experiment_name='single_model-single_attack-multiple_dataset')
    data.append(["Train", clean_accuracy, robust_accuracy,
                 model_robust_accuracy])

    clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_3(diff_model_name="e3_model",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks[0],
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=test_dataset_classes,
                                                                          experiment_name="single_model-single_attack-multiple_dataset",
                                                                          train=False)
    data.append(["Test", clean_accuracy, robust_accuracy,
                 model_robust_accuracy])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_3.csv", index=False)
    return df


def run_experiment4():
    # model mixture train and test
    target_model_archs = [(resnet18, "18"), (resnet34, "34")]
    test_model_archs = [(resnet50, "50")]
    train_dataset_classes = [MNIST]
    attacks = [ATTACKS['pgd_l2']]
    columns = ["config", "clean_accuracy", "robust_accuracy", "model_robust_accuracy"]
    data = []
    clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_3(diff_model_name=f"e4_model",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks[0],
                                                                          test_model_archs=target_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          experiment_name="multiple_model-single_attack-single_dataset",
                                                                          test_dataset_classes=train_dataset_classes)
    data.append(["Train", clean_accuracy, robust_accuracy,
                 model_robust_accuracy])

    clean_accuracy, robust_accuracy, model_robust_accuracy = experiment_3(diff_model_name="e4_model",
                                                                          train_model_archs=target_model_archs,
                                                                          attack=attacks[0],
                                                                          test_model_archs=test_model_archs,
                                                                          train_dataset_classes=train_dataset_classes,
                                                                          test_dataset_classes=train_dataset_classes,
                                                                          experiment_name="multiple_model-single_attack-single_dataset",
                                                                          train=False)
    data.append(["Test", clean_accuracy, robust_accuracy,
                 model_robust_accuracy])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"figures/experiment_4.csv", index=False)
    return df


if __name__ == "__main__":
    run_experiment1()