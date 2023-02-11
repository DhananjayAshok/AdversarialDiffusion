from torchattacks import PGD, PGDL2

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
