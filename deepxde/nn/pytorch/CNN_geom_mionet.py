import torch
import torch.nn as nn
import deepxde as dde
from deepxde.nn import NN
import sys

class CNN_GeomMIONet(NN):
    def __init__(
        self,
        layer_sizes_branch_1,
        layer_sizes_branch_2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super(CNN_GeomMIONet, self).__init__()
        
        self.CNN = layer_sizes_branch_1
        
        self.geoNet = layer_sizes_branch_2[0]
        self.geoNet2 = layer_sizes_branch_2[1]
        
        self.outNet = layer_sizes_trunk[0]
        self.outNet2 = layer_sizes_trunk[1]

        self.bias = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularization = regularization

    def forward(self, inputs):
        image_input = inputs[0]  # [bs, 1, 100, 100]
        params_input = inputs[1]  # [bs, 2], implicit geometry
        coord_input = inputs[2]  # [Npt, 4], output coordinates + implicit distance

        # First layers
        image_features = self.CNN(image_input)  # [bs, hidden dim] 
        param_features = self.geoNet(params_input)  # Output: [bs, hidden dim]
        coord_features = self.outNet(coord_input)  # Output: [Npt, hidden dim]

        # Mix between branch and trunk
        mixed_branch = param_features * torch.mean(coord_features, dim=0)  # Output: [bs, hidden dim]
        mixed_trunk = coord_features * torch.mean(param_features, dim=0)  # Output: [Npt, hidden dim]

        # Second layers
        param_encoded = self.geoNet2(mixed_branch)  # Output: [bs, hidden dim]
        coord_encoded = self.outNet2(mixed_trunk)  # Output: [Npt, hidden dim]

        # Joining branches
        branches = param_encoded + image_features
        
        # Output
        output = torch.einsum("bh,nh->bn", branches, coord_encoded) # Output: [bs, Npt]

        output += self.bias

        return output
