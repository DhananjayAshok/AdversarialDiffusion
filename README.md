# AdversarialDiffusion
Diffusion Models for Adversarial Attack Generation

## 1. Getting started.

Pip install the requirements

```matplotlib==3.5.1
pandas
torch==1.13.1
torchattacks==3.3.0
torchvision==0.14.1
tqdm==4.62.3
```

`pip install -r requirements.txt`

Download the ResNet trained checkpoints for every dataset from https://drive.google.com/file/d/1SowQ-KSCwM9lhUhIyaIWyJGH6GWssGo0/view . Copy the `ResNet_18`, `ResNet_18` and `ResNet_18` inside the zip into a new directory under repo root `models/`.

## 2. Running Experiments

Our code is easily reproducible. The experiments can simply be triggered from `experiments.py` by modifying the main code to run the desired experiment as `run_experiment{N}()` where `N = {1, 2, 3, 4}`.

Results of the experiments are dumped under `figures/experiment_{N}.csv` directory. It will include the fields `[dataset,attack,target_model,clean_accuracy,robust_accuracy,model_robust_accuracy,l2_norm,linf_norm]`. The rows pertain to some combination of *Target Architecture, Attack set and Datasets* and we record the metrics corresponding to these experiments.

Also trained models get created under `models/`.

### Experiment 1: Simplest setting
Single target model, single attack on single dataset. We run it for 4 combos.

````target_model_archs = [(resnet18, "18")]
attacks = [ATTACKS["fgsm"], ATTACKS['pgd']]
dataset_classes = [KMNIST, MNIST]
````

### Experiment 2: Multiple Attacks
Single target model, fused attack set and run on different datasets. We run it for the below combos.

```target_model_archs = [(resnet18, "18")]
attacks = [ATTACKS['pgd'], ATTACKS["fgsm"]]
dataset_classes = [MNIST, KMNIST]
```

### Experiment 3: Dataset Transfer
Measure how well can a diffusion attack trained on some datasets generalize to some unseen data, run for PGD.

```target_model_archs = [(resnet18, "18")]
train_dataset_classes = [MNIST, KMNIST]
test_dataset_classes = [FashionMNIST]
attacks = [ATTACKS['pgd']]
```

### Experiment 4: Model Transfer
Here the goal is to understand how accurately a diffusion attack learnt for some target model architectures work against another target architecture, checked with PGD.

```target_model_archs = [(resnet18, "18"), (resnet34, "34")]
test_model_archs = [(resnet50, "50")]
train_dataset_classes = [MNIST]
attacks = [ATTACKS['pgd']]
```
