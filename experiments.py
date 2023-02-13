from attacks import ImageAttack, AttackSet
from datasets import *


def get_common(target_model_archs, attacks, dataset_classes):
    models = []
    preprocessings = []
    for model in target_model_archs:
        for dset_class in dataset_classes:
            m, p = get_model(model, dset_class)
            models.append(m)
            preprocessings.append(p)
    attack_set = AttackSet(attacks, models, preprocessings)
    # mixture_dset = Dataset(components=[dataset_class]) or binary indicator
    return model, attack_set, mixture_dset


def experiment_1(target_model_arch, attack, dataset_class, experiment_name, train=False):
    model, attack_set, mixture_dset = get_common([target_model_arch], [attack], [dataset_class])    
    if train:
        #attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
        pass
    else:
        #attack_model = load_diffusion(experiment_name)
        pass
    #attack_success = measure_attack_success(model, mixture_dset, attack)
    #attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model)
    print(attack_success, attack_model_success)
    return attack_success, attack_model_success
  
def experiment_2(target_model_arch, attacks, dataset_class, experiment_name, train=False):
      model, attack_set, mixture_dset = get_common([target_model_arch], attacks, [dataset_class])    
    if train:
        #attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
        pass
    else:
        #attack_model = load_diffusion(experiment_name)
        pass
    attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model)
    attack_results = []
    for attack in attacks:
        #attack_success = measure_attack_success(model, mixture_dset, attack)
        attack_results.append(attack_success)
    return attack_results, attack_model_success
  
  
def experiment_3(target_model_archs, attack, dataset_classes, experiment_name, train=False):
    model, attack_set, mixture_dset = get_common(target_model_archs, attack, dataset_classes)    
    if train:
        #attack_model = get_diffusion(mixture_dset, attack_set, save_name=experiment_name)
        pass
    else:
        #attack_model = load_diffusion(experiment_name)
        pass
    attack_model_success = measure_attack_model_success(model, mixture_dset, attack_model)
    attack_results = []
    for dataset_class in dataset_classes:
        #dset = Dataset(components=[dset])
        #attack_success = measure_attack_success(model, dset, attack)
        attack_results.append(attack_success)
    return attack_results, attack_model_success
