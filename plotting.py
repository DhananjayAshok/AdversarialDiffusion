import matplotlib.pyplot as plt
import os
import torch
import pdb

models = ['simple_diffnet_ResNet18_KMNIST_FGSM',
'simple_diffnet_ResNet18_KMNIST_PGD',
'simple_diffnet_ResNet18_MNIST_FGSM',
'simple_diffnet_ResNet18_MNIST_PGD'
]
# for model_name in models:
#     path = os.path.join('/home/mmml/MMML/AdversarialDiffusion/models/', model_name, 'final_chkpt.pt')
#     ckpt = torch.load(path)
#     logs = ckpt['logs']
#     for k,
#     pdb.set_trace()
# plt.figure()


def plotting(X, X_correct, advs, advs_correct, model_attacked, model_attacked_correct, filename):
    indices = torch.randint(0, len(advs), size = (10,))
    X = X[indices]
    advs = advs[indices]
    model_attacked = model_attacked[indices]
    X_correct = X_correct[indices]
    advs_correct = advs_correct[indices]
    model_attacked_correct = model_attacked_correct[indices]
    fig, axes = plt.subplots(nrows =3, ncols = 10, figsize = (3 * 8, 9))
    for i in range(10):
        # display the image X[i]
        axes[0][i].imshow(X[i][0], cmap = 'gray')
        # if i == 0:
        axes[0][i].set_title(f'Original\nPrediction: {X_correct[i]}', fontsize = 16)
        axes[0][i].axis('off')

        # display the adversarial example advs[i]
        axes[1][i].imshow(advs[i][0], cmap = 'gray')
        # if i == 0:
        axes[1][i].set_title(f'Adversarial\nPrediction: {advs_correct[i]}', fontsize = 16)
        axes[1][i].axis('off')

        # display the attacked image
        axes[2][i].imshow(model_attacked[i][0], cmap = 'gray')
        # if i == 0:
        axes[2][i].set_title(f'Diffusion\nPrediction: {model_attacked_correct[i]}', fontsize = 16)
        axes[2][i].axis('off')

    fig.subplots_adjust(wspace=0., hspace = 0.3)
    # pdb.set_trace()
    fig.savefig(filename, bbox_inches='tight')