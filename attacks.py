from torchattacks import PGD, PGDL2
from numpy.random import choice

ATTACKS = {"pgd": PGD, "pdg_l2": PGDL2}


class ImageAttack:
    def __init__(self, attack_class, params=None):
        self.params = params
        self.attack_class = attack_class
        if params is None:
            self.params = {"eps": 0.001}

    def __call__(
        self, model, input_batch, true_labels, target_labels=None, preprocessing=None
    ):
        attack = self.attack_class(model, **self.params)
        attack.set_normalization_used(mean=list(preprocessing['mean']), std=list(preprocessing['std']))
        return attack(input_batch, true_labels)

    
class AttackSet:
    def __init__(self, attacks, models, preprocessings):
        self.attacks = attacks
        self.models = models
        self.preprocessings = preprocessings
        self.n_models = len(models)
        assert self.n_models == len(self.preprocessings)
        
    def sample(self, input_batch, true_labels):
        i = choice(range(self.n_models))
        model = self.models[i]
        preprocessing = self.preprocessings[i]
        attack = choice(self.attacks)
        return attack(model, input_batch, true_labels, preprocessing=preprocessing)
        
        
