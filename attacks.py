from torchattacks import PGD, PGDL2
from numpy.random import choice

ATTACKS = {"pgd": PGD, "pgd_l2": PGDL2}


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
    def __init__(self, attacks, params=None):
        self.attacks = attacks
        self.params = params

    def __call__(self, model, input_batch, true_labels, preprocessing=None):
        attack = ImageAttack(choice(self.attacks), self.params)
        return attack(model, input_batch, true_labels, preprocessing=preprocessing)


if __name__ == "__main__":
    pass
