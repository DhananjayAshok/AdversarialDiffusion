import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

from utils import safe_mkdir, Parameters, normalize_to_dict

data_root = "data/"
NUM_LABELS = {ds.CIFAR10: 10, ds.Caltech101: 101, ds.MNIST: 10, ds.FashionMNIST: 10, ds.KMNIST: 10}


def get_torchvision_dataset(dataset_class, train=False):
    transform = None
    if dataset_class == ds.CIFAR10:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
    if dataset_class == ds.Caltech101:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    # if dataset_class == ds.Food101:
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(size=(224, 224)), # TODO: I think 512
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ]
    #     )

    if dataset_class == ds.MNIST:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.127,), std=(0.2959,))
            ]
        )

    # IDK about the mean std of all these.

    if dataset_class == ds.FashionMNIST:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.127,), std=(0.2959,))
            ]
        )

    if dataset_class == ds.KMNIST:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.127,), std=(0.2959,))
            ]
        )

    # if dataset_class == ds.FER2013:
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(size=(48, 48)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.127,), std=(0.2959,))
    #         ]
    #     )

    save_path = data_root + f"/{dataset_class.__name__}/"
    safe_mkdir(save_path)
    if not dataset_class == ds.Caltech101:
        return dataset_class(save_path, transform=transform, download=True, train=train)
    else:
        return dataset_class(save_path, transform=transform, download=True)


def sample_torch_dataset(dset, batch_size=32, shuffle=False):
    f, h = dset[0]
    X_shape = f.shape
    # We assume y is a scalar output
    batch_shape = (batch_size,) + X_shape
    y_shape = (batch_size,)
    device = f.device
    X = torch.zeros(size=batch_shape).to(device)
    y = torch.zeros(size=y_shape).to(device)
    if shuffle:
        idx = torch.randint(low=0, high=len(dset), size=(batch_size,))
    else:
        idx = range(batch_size)
    for i, id in enumerate(idx):
        X[i], y[i] = dset[id]
    return X, y


def get_torchvision_dataset_sample(dataset_class, train=False, batch_size=32):
    dset = get_torchvision_dataset(dataset_class, train=train)
    X, y = sample_torch_dataset(dset, batch_size=batch_size, shuffle=True)
    X = X.to(Parameters.device)
    y = y.to(Parameters.device)
    return X, y.long()


class DatasetAndModels(torch.utils.data.Dataset):

    def __init__(self, dataset_classes, model_list, train=True, force_balance=False):
        """

        :param dataset_classes: list of torchvision.dataset classes
        :param model_list: nested list of trained models which have been trained for the dataset
        :param train: split of data
        """
        assert len(dataset_classes) > 0
        self.datasets = []
        self.preprocessings = []
        for component in dataset_classes:
            dataset = get_torchvision_dataset(component, train=train)
            self.datasets.append(dataset)
            if len(dataset.transform.transforms) < 3:
                transform = transforms.Normalize(mean=(0,), std=(1,))
            else:
                transform = dataset.transform.transforms[2] # Hard code normalize on 2
            self.preprocessings.append(normalize_to_dict(transform))
        self.lengths = [len(d) for d in self.datasets]
        if force_balance:
            tmp_datasets = self.datasets
            self.datasets = []
            min_length = min(self.lengths)
            for dataset in tmp_datasets:
                if len(dataset) == min_length:
                    self.datasets.append(dataset)
                else:
                    rand_indices = np.arange(len(dataset))
                    np.random.shuffle(rand_indices)
                    new_indices = rand_indices[:min_length]
                    self.datasets.append(Subset(dataset, new_indices))
            self.lengths = [min_length for d in range(len(self.datasets))]
        self.models = model_list
        assert len(self.models) == len(self.datasets)
        for element in model_list:
            assert len(element) > 0, "Maybe you forgot to make model list a nested list"
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        """

        :param index: iteration index
        :return: subindex i which corresponds to the dataset list (self.models[i] gives eligible models for dataset[i],
        similarly for self.preprocessings
        """
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return i, self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length


def get_sample_mixture_dset():
    from models import get_model
    from torchvision.models import resnet18
    return DatasetAndModels(dataset_classes=[ds.MNIST], model_list=[[get_model(model=resnet18(),dset_class=ds.MNIST,
                                                                              save_suffix="18")]])