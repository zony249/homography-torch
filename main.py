import os 
import numpy as np 
from typing import Tuple, List, Optional, Union, Literal
from PIL import Image

import torch 
from torch import nn 
import torch.nn.functional as F
from torch.optim import AdamW

import matplotlib.pyplot as plt 

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class Homography(nn.Module): 
    def __init__(self): 
        super().__init__()  
        self.basis = torch.zeros((8, 3, 3), device=DEVICE)
        self.basis[0,0,2] = 1. 
        self.basis[1,1,2] = 1. 
        self.basis[2,0,1] = 1. 
        self.basis[3,1,0] = 1.
        self.basis[4,0,0], self.basis[4,1,1] = 1., -1. 
        self.basis[5,1,1], self.basis[5,2,2] = -1., 1.
        self.basis[6,2,0] = 1. 
        self.basis[7,2,1] = 1. 

        self.v = nn.Parameter(torch.zeros((8,1,1), device=DEVICE)*0.1, requires_grad=True) 
        self.conv = nn.Conv2d(1, 1, 3, padding="same").to(DEVICE)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        I: torch.Tensor[B, C, H, W]
        """
        H = self.Mexp(self.basis, self.v) 
        
        h, w = I.shape[-2], I.shape[-1]
        x = torch.linspace(-1, 1, w, device=DEVICE)
        y = torch.linspace(-1, 1, h, device=DEVICE) 
        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).T
        grid_t = H @ torch.cat([grid, torch.ones_like(grid[-1:, :], device=DEVICE)], dim=0)
        grid_t /= grid_t[2:, :].clone()
        xx_t, yy_t = grid_t[0, :].reshape(xx.shape), grid_t[1, :].reshape(yy.shape)

        grid_sample = torch.stack([xx_t, yy_t], dim=-1)[None, :, :, :]
        J = F.grid_sample(I, grid_sample, align_corners=False)
        return J, H 
    
    def Mexp(self, B, v): 
        A = torch.eye(3, device=DEVICE)
        n_fact = 1
        H = torch.eye(3, device=DEVICE)
        for i in range(11): 
            A = (v * B).sum(dim=0) @ A
            n_fact = max(i, 1) * n_fact
            A /= n_fact
            H += A 
        return H / H[2, 2]

    def log_factorial(self, x):
        return torch.lgamma(x + 1)
    def factorial(self, x): 
        return torch.exp(self.log_factorial(x))




class Trainer: 
    def __init__(self, 
                 Hnet, 
                 lr=1e-2,
                 levels=3,
                 steps_per_epoch=10, 
                 loss_fn=None): 
        self.Hnet = Hnet
        self.optim = AdamW(self.Hnet.parameters(), lr=lr)
        self.levels = levels 
        self.steps_per_epoch = steps_per_epoch
        self.loss_fn = loss_fn
        if self.loss_fn is None: 
            self.loss_fn = mse


    def convert_img_to_torch(self, imgI: np.ndarray) -> torch.Tensor: 

        if imgI.ndim == 2: 
            I = torch.from_numpy(imgI).float().to(DEVICE)[None, None, :, :]
        elif imgI.ndim == 3:  
            I = torch.permute(torch.from_numpy(imgI), 
                            (2, 0, 1)).float().to(DEVICE)[None, :, :, :]
        return I 

    def register(self, imgI: np.ndarray, imgJ: np.ndarray): 

        I = self.convert_img_to_torch(imgI)
        J = self.convert_img_to_torch(imgJ)
        with torch.no_grad(): 
            J_w, H = self.Hnet(J)
            pre_reg_J = J_w.detach().cpu().numpy()[0, 0]
            plt.imshow(pre_reg_J)
            plt.savefig("pre-registration-J.png")

        scales = 2.0**torch.arange(-self.levels, 1, device=DEVICE)


        for level in range(1, self.levels + 1): 

            I_s, J_s = self.scale(I, scales[level]), self.scale(J, scales[level])
            scale_J = J_s.detach().cpu().numpy()[0, 0]
            plt.imshow(scale_J)
            plt.savefig(f"scale_{scales[level]}--{level}.png")

            nn.init.zeros_(self.Hnet.conv.weight)
            nn.init.zeros_(self.Hnet.conv.bias)

            for step in range(self.steps_per_epoch): 
                self.Hnet.zero_grad()                
                J_w, H = self.Hnet(J_s)
                J_w = self.Hnet.conv(J_w) + J_w
                loss = self.loss_fn(I_s.flatten(), J_w.flatten())
                loss.backward() 
                self.optim.step() 
                print(loss.item())
            

        return J_w, H

    def scale(self, I, s): 


        k_x = torch.linspace(-1/s, 1/s, int(2/s - 1), device=DEVICE) 
        xx, yy = torch.meshgrid(k_x, k_x, indexing="xy")
        kernel = torch.exp(-0.5 * (xx**2 + yy**2)/s**2)
        kernel /= kernel.sum()
        kernel = kernel[None, None, :, :]
        kernel += torch.zeros((I.shape[1], 1, 1, 1), device=DEVICE)
        I_smooth = F.conv2d(I, kernel, groups=I.shape[1]) 


        h, w = I.shape[-1], I.shape[-2]

        x = torch.linspace(-1, 1, int(s*w), device=DEVICE)
        y = torch.linspace(-1, 1, int(s*h), device=DEVICE) 

        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        grid_s = torch.stack([xx, yy], dim=-1)[None, :, :, :]

        Is = F.grid_sample(I_smooth, grid_s, align_corners=False)

        return Is



"""
This code was obtained from https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3. 
"""
class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=DEVICE).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=DEVICE).float() + 0.5)

    def forward(self, x, y):
        x = (torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1))[:, ::100]
        y = (torch.unsqueeze(y, 0) - torch.unsqueeze(self.centers, 1))[:, ::100]

        xy = (x[:, None, :]**2 + y[None, :, :]**2)

        hist = torch.exp(-0.5*(xy)/self.sigma**2) / (self.sigma**2 * np.pi*2) * self.delta
        hist = hist.sum(dim=-1)
        return hist * 100


def mse(targets: torch.Tensor, inputs: torch.Tensor): 
    mse_loss = F.mse_loss(inputs, targets) 
    return mse_loss 

def histogram_mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


if __name__ == "__main__": 
    
    Hnet = Homography().to(DEVICE) 
    
    # I = torch.zeros((1, 1, 5, 6), device=DEVICE)

    imgI = Image.open("knee1.bmp")
    imgI = np.array(imgI) / 255.
    imgJ = Image.open("knee2.bmp")
    imgJ = np.array(imgJ) / 255.
    
    plt.imshow(imgI)
    plt.savefig("imgI.png")

    plt.imshow(imgJ)
    plt.savefig("imgJ.png")

    Hnet = Homography() 




    trainer = Trainer(
        Hnet, 
        lr=1e-3, 
        levels=5, 
        steps_per_epoch=1000, 
        loss_fn=mse, 
    )
    J, H = trainer.register(imgI, imgJ) 

    registered_J = J.detach().cpu().numpy()[0, 0]
    plt.imshow(registered_J)
    plt.savefig("registered_J.png")


    pre_mi = histogram_mutual_information(image1=imgI, image2=imgJ)
    post_mi = histogram_mutual_information(image1=imgI, image2=registered_J)

    print(f"MI before registering: {pre_mi}")
    print(f"MI after registering: {post_mi}")
