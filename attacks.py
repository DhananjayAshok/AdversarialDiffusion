from torchattacks import PGD, PGDL2, FGSM, TIFGSM
from numpy.random import choice

ATTACKS = {"pgd": PGD, "pgd_l2": PGDL2, "fgsm": FGSM, "tifgsm": TIFGSM}


class ImageAttack:
    def __init__(self, attack_class, params=None):
        self.params = params
        self.attack_class = attack_class
        if params is None:
            self.params = {"eps": 1.0}

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
        cand = choice(self.attacks)
        if isinstance(cand, ImageAttack):
            attack = cand
        else:
            attack = ImageAttack(cand, self.params)
        return attack(model, input_batch, true_labels, preprocessing=preprocessing)


if __name__ == "__main__":
    pass
