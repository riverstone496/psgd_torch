
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['mlp']

class MLP(nn.Module):
    def __init__(self, n_hid=1000, nonlin=torch.relu, img_size=28, num_classes=10, num_channels=1, bias=False, depth=3):
        super().__init__()
        self.img_size=img_size
        self.num_channels=num_channels
        self.nonlin=nonlin
        self.depth = depth

        self.input_layer = nn.Linear(img_size*img_size*num_channels, n_hid,bias=bias)
        self.hidden_layer = nn.Linear(n_hid, n_hid,bias=bias)
        self.output_layer = nn.Linear(n_hid, num_classes,bias=bias)

        self.input_layer.base_fan_in = img_size*img_size*num_channels
        self.hidden_layer.base_fan_in = 64
        self.output_layer.base_fan_in = 64
        self.input_layer.base_fan_out = 64
        self.hidden_layer.base_fan_out = 64
        self.output_layer.base_fan_out = num_classes
        self.num_features = n_hid

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.nonlin(self.input_layer(x))
        if self.depth!=2:
            x = self.nonlin(self.hidden_layer(x))
        x = self.output_layer(x)
        return x