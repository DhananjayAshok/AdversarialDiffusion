from models import get_model
from attacks import AttackSet, ATTACKS
from datasets import DatasetAndModels
from utils import get_common
import torch


def test_1():
    from torchvision.models import resnet18, resnet34
    from torchvision.datasets import MNIST, FashionMNIST, KMNIST
    model_archs = [(resnet18, "18"), (resnet34, "34")]
    attack = [ATTACKS['pgd_l2']]
    attack_set, mixture_dset = get_common(model_archs, attack, [MNIST, FashionMNIST, KMNIST], train=True)
    return attack_set, mixture_dset


if __name__ == "__main__":
    attack_set, mixture_dset = test_1()
    i, data = mixture_dset[12203]
    model = mixture_dset.models[i][0]
    preprocessing = mixture_dset.preprocessings[i]
    X, y = data
    X = X.reshape(1, 1, 28, 28)
    y = torch.tensor([y])
    adv_image = attack_set(model, X, y, preprocessing)


