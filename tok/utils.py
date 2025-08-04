import torch
import torch.nn as nn


class ScalingLayer(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        self.register_buffer('shift', torch.Tensor(mean)[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(std)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    
    def inv(self, inp):
        return inp * self.scale + self.shift