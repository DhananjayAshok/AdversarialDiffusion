import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class DDPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = 64,
            timesteps = 1000,   # number of steps
            loss_type = 'l1'    # L1 or L2
        )

        # training_images = torch.rand(4, 3, 128, 128) # images are normalized from 0 to 1

    def forward(self, x):
        batch, nc, h, w = x.shape
        x = x.expand(batch, 3, h, w)
        _, out = self.diffusion(x)
        out = out.mean(dim=1, keepdims=True)
        return out

if __name__ == '__main__':
    training_images = torch.rand(4, 3, 128, 128)
    model = DDPM()
    out = model(training_images)
