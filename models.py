import torch
from torch import nn
from torchvision.models import resnet50, resnet18, resnet34
from torch.utils.data import Subset
import numpy as np
import os
from tqdm import tqdm
from datasets import get_torchvision_dataset
from utils import Parameters,safe_mkdir

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_model(model, dset_class, save_suffix, n_classes=10, input_channels=1):
    modify_network(model, n_classes, input_channels)
    if save_suffix != "":
        save_suffix = f"_{save_suffix}"
    model_path = os.path.join(Parameters.model_path, model.__class__.__name__+save_suffix,
                              dset_class.__name__, "final_chkpt.pt")
    state_dict = torch.load(model_path, map_location= device)
    model.load_state_dict(state_dict)
    return model

def modify_network(model, n_classes, input_channels):
    model_type = model.__class__.__name__
    if not input_channels == 3:
        if model_type == "ResNet":
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif model_type == "AlexNet":
            model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        elif model_type == "VGG":
            model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if not n_classes == 1000:
        if model_type == "ResNet":
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif model_type == "AlexNet":
            model.classifier[6] = nn.Linear(4096, n_classes)
        elif model_type == "VGG":
            model.classifier[6] = nn.Linear(4096, n_classes)


def train(model, dataset_class, n_classes=10, input_channels=1, num_epochs=5, batch_size=32, lr=0.001, train_size=0.9,
          early_stopping=1,
          save_suffix=""):
    device = Parameters.device
    whole_train_dataset = get_torchvision_dataset(dataset_class, train=True)
    train_size_int = int(len(whole_train_dataset) * train_size)
    rand_indices = np.arange(len(whole_train_dataset))
    np.random.shuffle(rand_indices)
    train_indices = rand_indices[:train_size_int]
    val_indices = rand_indices[train_size_int:]
    train_dataset = Subset(whole_train_dataset, train_indices)
    val_dataset = Subset(whole_train_dataset, val_indices)
    test_dataset = get_torchvision_dataset(dataset_class, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size=batch_size
        , shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset
        , batch_size=batch_size
        , shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size=batch_size
        , shuffle=True)
    n_total_step = len(train_loader)
    modify_network(model, n_classes=n_classes, input_channels=input_channels)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    if save_suffix != "":
        save_suffix = f"_{save_suffix}"
    checkpoint_path = os.path.join(Parameters.model_path, model.__class__.__name__+save_suffix,
                                   dataset_class.__name__)
    final_path = os.path.join(Parameters.model_path, model.__class__.__name__+save_suffix, dataset_class.__name__,
                              "final_chkpt.pt")
    safe_mkdir(checkpoint_path)
    best_val_loss = np.inf
    epochs_without_improvement = 0
    print(f"Starting Training for {dataset_class.__name__} on {model.__class__.__name__+save_suffix}")
    for epoch in range(num_epochs):
        for i, (imgs , labels) in tqdm(enumerate(train_loader), total=n_total_step):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            if (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, '
                      f'acc = {100*(n_corrects/labels.size(0)):.2f}')
            if (i+1) % 500 == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_{epoch}.pth"))
            del imgs
            del labels
            del labels_hat
        scheduler.step()
        val_loss = _val(model, val_loader, criterion)
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
        if early_stopping is not None:
            if epochs_without_improvement >= early_stopping:
                print(f"Early Stopping...")
                break

    torch.save(model.state_dict(), final_path)
    test_loss = _val(model, test_loader, criterion)
    return test_loss


def _val(model, test_loader, loss):
    device = Parameters.device
    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        val_losses = []
        for i, (test_images_set, test_labels_set) in enumerate(test_loader):
            test_images_set = test_images_set.to(device)
            test_labels_set = test_labels_set.to(device)

            y_predicted = model(test_images_set)
            labels_predicted = y_predicted.argmax(axis=1)
            loss_value = loss(y_predicted, test_labels_set)
            number_corrects += (labels_predicted == test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
            val_losses.append(loss_value)
        print(f'Overall accuracy: {(number_corrects / number_samples) * 100} % Loss: {loss_value}')
        return sum(val_losses)/len(val_losses)


if __name__ == "__main__":
    from torchvision.datasets import MNIST, FashionMNIST, KMNIST
    model = resnet18()
    model1 = resnet34()
    model2 = resnet50()
    models = [(model, "18"), (model1, "34"), (model2, "50")]
    datasets = [MNIST, FashionMNIST, KMNIST]
    for dataset_class in datasets:
        for model, save_suffix in models:
            train(model, dataset_class, save_suffix=save_suffix)
