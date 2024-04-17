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

        dims = [latent_size + 6] + [512, 512, 512, 512, 512, 512, 512, 512, 512] + [last_dim]
        self.output_dim = last_dim
        self.num_layers = len(dims)
        self.latent_in = [4,7]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            if l < self.num_layers - 2:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, inputxyz, inputDataDims=3):
        xyz   = inputxyz[:, -6:-3]
        direc = inputxyz[:,-3:]
        

        #perpendicularfoot
        f = torch.cross(direc,torch.cross(xyz, direc,dim=-1),dim=-1)

        
        #distance to perpendicular foot
        d = ((f-xyz)*direc).sum(-1)

        input = torch.cat([inputxyz[:,:-6],f, direc], 1)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)



        return x[:,0]+d, self.sigm(x[:,1])

