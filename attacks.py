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
        self, model, input_batch, true_labels, target_labels=None, preprocessing=None):
        attack = self.attack_class(model, **self.params)
        attack.set_normalization_used(mean=list(preprocessing['mean']), std=list(preprocessing['std']))
        return attack(input_batch, true_labels)

    
class AttackSet:
    def __init__(self, attacks):
        self.attacks = attacks
        
    def __call__(self, model, input_batch, true_labels, preprocessing=None):
        attack = choice(self.attacks)
        return attack(model, input_batch, true_labels, preprocessing=preprocessing)
        
        
