from .runners.diffpure_ddpm import Diffusion
from .runners.diffpure_guided import GuidedDiffusion
# from runners.diffpure_sde import RevGuidedDiffusion
# from runners.diffpure_ode import OdeGuidedDiffusion
# from runners.diffpure_ldsde import LDGuidedDiffusion

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

import time
import argparse
import yaml
import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class DiffPure(torch.nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        # elif args.diffusion_type == 'sde':
        #     self.runner = RevGuidedDiffusion(args, config, device=config.device)
        # elif args.diffusion_type == 'ode':
        #     self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        # elif args.diffusion_type == 'ldsde':
        #     self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        batch, nc, h, w = x.shape
        x = x.expand(batch, 3, h, w)
        # if counter % 5 == 0:
        #     print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        # end_time = time.time()
        # print('image editing sample: ', end_time - start_time, ' s')
        # pdb.set_trace()
        # minutes, seconds = divmod(time.time() - start_time, 60)

        # s = time.time()
        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
        # e = time.time()

        # if counter % 5 == 0:
        #     print(f'x shape (before diffusion models): {x.shape}')
        #     print(f'x shape (before classifier): {x_re.shape}')
        #     print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = (x_re + 1) * 0.5

        out = out.mean(dim=1, keepdims=True)

        return out

# Define a function to recursively convert a dictionary to a namespace object
def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, dict_to_namespace(v))
        else:
            setattr(namespace, k, v)
    return namespace

# from unet import SimpleUnet
if __name__ == '__main__':
    config_file = 'configs/guided_diffusion_config.yaml'
    # Load YAML file into a Python dictionary
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Convert the dictionary to a namespace object
    config_namespace = dict_to_namespace(config_dict)

    args = config_namespace.args
    # args.log_dir = 'logs'

    config = config_namespace.config
    # config.device = device
    # config.image_size = 64
    # config.num_channels = 3

    diffpure = DiffPure(args, config)

    image = torch.from_numpy(np.random.randint(0, 255, size=(1, 3, 64, 64)).astype(np.uint8))
    out = diffpure(image)

    from torchsummary import summary
    summary(model= diffpure, input_size=(3, 64, 64))

    print(out)
