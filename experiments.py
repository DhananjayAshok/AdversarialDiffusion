from attacks import ImageAttack, ATTACKS
from torchvision.models import resnet50
from torchvision.datasets import MNIST
from utils import measure_attack_success, measure_attack_model_success, get_common


# TODO: add code the attack model.
def get_diffusion(mixture_dset, attack_set, save_name=""):
    return lambda x: x


def load_diffusion(save_name):
    return lambda x: x


def experiment_1(target_model_arch, attack, dataset_class, experiment_name = 'single', train=False):
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
        attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_success = measure_attack_success(model, mixture_dset, attack)
    attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model, model=model)
    return attack_success, attack_model_success


def experiment_2(target_model_arch, attacks, dataset_class, experiment_name, train=False):
    if train:
        attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class], train=True)
        attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
    else:
        attack_model = load_diffusion(experiment_name)
    attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class], train=False)
    model = mixture_dset.models[0][0]
    attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model)
    attack_results = []
    for attack in attacks:
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
    train = False
    a, b = experiment_1(target_model_arch, attack, dataset_class, experiment_name, train=train)
    print(a)
    print(b)
    return a, b


if __name__ == "__main__":
    pass
