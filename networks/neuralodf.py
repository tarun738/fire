#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

'''
Ported from DeepSDF
https://github.com/facebookresearch/DeepSDF
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))

class DDFNet(nn.Module):
    def __init__(
        self,
        latent_size,
        last_dim=2
    ):
        super(DDFNet, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 6] + [512, 512, 512, 512, 512, 512, 512] + [last_dim]
        self.output_dim = 2
        self.num_layers = len(dims)
        self.latent_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, input, inputDataDims=3):
       
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)



        return x[:,0], self.sigm(x[:,1])
