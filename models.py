import torch
from torch import nn
from torchvision.models import resnet50, vgg19_bn, inception_v3
import os
import tqdm
from datasets import get_torchvision_dataset
import utils

# TODO: CHECK ARCH WORKS THIS WAY AND ISNT A FUNCTION
def get_model(model_arch, dset_class):
    model_path = os.path.join(utils.Parameters.model_path, model_arch, dset_class, "final_chkpt.pt")
    model = torch.load(model_path)
    return model


def train(model, dataset_class, n_classes=10, input_channels=3, num_epochs=20, batch_size=32, lr=0.001, save_suffix=""):
    device = utils.Parameters.device
    train_dataset = get_torchvision_dataset(dataset_class, train=True)
    test_dataset = get_torchvision_dataset(dataset_class, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size=batch_size
        , shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size=batch_size
        , shuffle=True)
    n_total_step = len(train_loader)
    print(n_total_step)
    if not input_channels == 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    if not n_classes == 1000:
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    checkpoint_path = os.path.join(utils.Parameters.model_path, model.__class__.__name__, dataset_class)
    final_path = os.path.join(utils.Parameters.model_path, model.__class__.__name__, dataset_class, "final_chkpt.pt") # TODO: Change

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
            n_correct_n = 0
            if (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}, aug acc = {100*(n_correct_n/labels.size(0)):.2f}%')
            if (i+1) % 500 == 0:
                torch.save(model, os.path.join(checkpoint_path, f"chkpt_{save_suffix}_{epoch}.pth"))
            del imgs
            del labels
            del labels_hat
        scheduler.step()
        val_loss = _val(model, test_loader, criterion)
    torch.save(model, final_path)


def _val(model, test_loader, loss):
    device = utils.Parameters.device
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
