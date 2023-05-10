# AdversarialDiffusion
Diffusion Models for Adversarial Attack Generation

### 1. Getting started.

Pip install the requirements

`matplotlib==3.5.1
pandas
torch==1.13.1
torchattacks==3.3.0
torchvision==0.14.1
tqdm==4.62.3`

```pip install -r requirements.txt```

### 2. Running Experiments

Our code is easily reproducible. The experiments can simply be triggered from `experiments.py` by modifying the main code to run the desired experiment as `run_experiment{N}()` where `N = {1, 2, 3, 4}`.

Results of the experiments are dumped under `figures/experiment_{N}.csv` directory. It will include the fields `[dataset,attack,target_model,clean_accuracy,robust_accuracy,model_robust_accuracy,l2_norm,linf_norm]`. The rows pertain to some combination of *Target Architecture, Attack set and Datasets* and we record the metrics corresponding to these experiments.


### Experiment 1: Simplest setting
### Experiment 2: Simplest setting
### Experiment 3: Simplest setting
### Experiment 4: Simplest setting