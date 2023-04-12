import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import pdb

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
        out, loss = self.diffusion(x)
        out = out.mean(dim=1, keepdims=True)
        # pdb.set_trace()
        return out

if __name__ == '__main__':
    # training_images = torch.rand(4, 3, 64, 64)
    # model = DDPM()
    # out = model(training_images)
    # print(out.shape)


    unet = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        unet,
        image_size = 64,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    for i in range(10):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.01) #, weight_decay=5e-4)

        torch.random.manual_seed(0)
        training_images = torch.rand(8, 3, 64, 64, requires_grad = False)
        actual = training_images.mean(dim = 1, keepdims = True)

        # pdb.set_trace()
        loss_value, model_out = diffusion(training_images)
        # loss_value = criterion(model_out, actual)
        loss_value.backward()

        optimizer.step()
        print(f'loss: {loss_value.item()} | params: {next(diffusion.parameters())[0][0][0][:5]}')
        pdb.set_trace()