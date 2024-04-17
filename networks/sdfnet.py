# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.posEncode import positional_encoding

class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize, indim, tvlsqrt,globalFeats=False):
        super().__init__()
        self.fsize = fsize
        self.globalFeats = globalFeats
        if self.globalFeats:
            self.fdim = fdim + 1
        else:
            self.fdim = fdim
        self.indim = indim
        self.tvlsqrt = tvlsqrt
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)

    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3) # [N, 1, 1, 3]
            sample = F.grid_sample(self.fm, sample_coords,
                                   align_corners=True, padding_mode='border')[0,:,:,0,0].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3) # [N, 1, 1, 3]
            sample = F.grid_sample(self.fm, sample_coords,
                                   align_corners=True, padding_mode='border')[0,:,:,:,0].permute([1,2,0])
        if self.globalFeats:
            globVal = sample[:,-1]
            sample = sample[:,:-1]
            return globVal, sample
        else:
            return sample
    def getTVL(self):
        g_w = (( self.fm[:,:,1:,1:,1:] - self.fm[:,:,:-1,1:,1:] )**2)
        g_h = (( self.fm[:,:,1:,1:,1:] - self.fm[:,:,1:,:-1,1:] )**2)
        g_l = (( self.fm[:,:,1:,1:,1:] - self.fm[:,:,1:,1:,:-1] )**2)
        loss = g_h+g_w+g_l
        if self.tvlsqrt:
            loss = (loss+1e-10)**0.5
        return loss.mean()
class FeaturePlanes(nn.Module):
    def __init__(self, fdim, fsize, indim, tvlsqrt,globalFeats=False, init_mul=1.0,l1=False):
        super().__init__()
        self.fsize = fsize
        self.globalFeats = globalFeats
        if self.globalFeats:
            self.fdim = fdim + 1
        else:
            self.fdim = fdim
        self.indim = indim
        self.tvlsqrt = tvlsqrt
        self.l1 = l1
        self.fm = nn.ParameterList()
        for i in range(0,self.indim-1):
            for j in range(i+1,self.indim):
                self.fm.append(nn.Parameter(torch.randn(1, self.fdim, self.fsize+1, self.fsize+1) * 0.01*init_mul))
        self.sparse = None
    def getFeats(self,x,idx):
        N = x.shape[0]
        sample_coords = x.reshape(1, N, 1, 2) # [N, 1, 1, 2]
        sample = F.grid_sample(self.fm[idx], sample_coords,
                                align_corners=True, padding_mode='border')[0,:,:,0].transpose(0,1)
        return sample

    def forward(self, x):
        samples = None
        idx = 0
        if self.globalFeats:
            globVal = 0
        for i in range(0,self.indim-1):
            for j in range(i+1,self.indim):
                tmp = self.getFeats(torch.cat([x[:,i].unsqueeze(-1),x[:,j].unsqueeze(-1)],dim=1),idx)
                if self.globalFeats:
                    globVal += tmp[:,-1]
                    tmp = tmp[:,:-1]
                if samples is None:
                    samples = tmp
                else:
                    samples = torch.cat([samples,tmp],dim=1)
                idx += 1
        if self.globalFeats:
            return samples, globVal
        else:
            return samples
    def getTVL(self):
        idx = 0
        tvl = 0
        for i in range(0,self.indim-1):
            for j in range(i+1,self.indim):
                tvl += self.computeTVL(self.fm[idx])
                idx += 1
        
        return tvl/len(self.fm)
    def computeTVL(self,feats):
        if self.l1:
            g_h = torch.abs(( feats[:,:,1:,1:] - feats[:,:,1:,:-1] ))
            g_w = torch.abs(( feats[:,:,1:,1:] - feats[:,:,:-1,1:] ))
            loss = g_h+g_w
            return loss.mean()
        else:
            g_h = (( feats[:,:,1:,1:] - feats[:,:,1:,:-1] )**2)
            g_w = (( feats[:,:,1:,1:] - feats[:,:,:-1,1:] )**2)
            loss = g_h+g_w
            if self.tvlsqrt:
                loss = (loss+1e-10)**0.5
            return loss.mean()

class SDFNet(nn.Module):
    def __init__(self, latent_size,
                       input_dim,
                       feature_dim,
                       feature_size,
                       hidden_dim,
                       pos_invariant,
                       num_hidden_layers=1,
                       positionalEncoding=0,
                       tvlsqrt = False,
                       global_feats = False,
                       splitLat=False,
                       featVol=False):

        super(SDFNet, self).__init__()
        self.fdim = feature_dim
        self.fsize = feature_size
        self.hidden_dim = hidden_dim
        self.splitLat = splitLat
        if self.splitLat:
            latent_size = latent_size//2
        self.latent_size = latent_size
        self.pos_invariant = pos_invariant
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.positionalEncoding = positionalEncoding
        self.hasColor = False
        self.global_feats = global_feats
        self.tvlsqrt = tvlsqrt
        self.featVol = featVol
        if not self.fdim == 0:
            if not self.featVol:
                self.features_planes = FeaturePlanes(self.fdim, self.fsize, self.input_dim, self.tvlsqrt, globalFeats=self.global_feats)
            else:
                self.features_planes = FeatureVolume(self.fdim, self.fsize, self.input_dim, self.tvlsqrt, globalFeats=self.global_feats)
            self.numplanes = len(self.features_planes.fm)
        else:
            self.numplanes = 0
            self.features_planes = None
        self.network = None

        # concatenate featureplane features, so n* the feature dimension
        self.sdf_input_dim = self.numplanes*self.fdim + self.latent_size

        # positional encoding, add the input positionally  encoded (or just xyz and/or dir) to the input dimension
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim + self.input_dim * 2 * self.positionalEncoding # if self.input_dim==3 else 4

        
        layers = []
        layers.append(nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True))
        layers.append(nn.ReLU())
        for lay in range(0,self.num_hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, 1, bias=True))
        self.network = nn.Sequential(*layers)


    def getFeatTVL(self):
        if self.features_planes is None:
            return 0
        else:
            return self.features_planes.getTVL()
        
    def forward(self,x, inputDataDims=3,addFeats=None, inglobfeats=False):
        # import pdb; pdb.set_trace()
        global_feats = self.global_feats if not inglobfeats else inglobfeats
        # import pdb; pdb.set_trace()
        if addFeats is not None:
            currFeatPL = addFeats
        else:
            currFeatPL = self.features_planes
        if self.input_dim == 3:
            inputDataDims = 3
            in_x = x[:,-3:]
            if global_feats:
                feats, globalVal = currFeatPL(x[:,-3:].detach())
            elif currFeatPL is not None:
                feats = currFeatPL(x[:,-3:].detach())
            else:
                feats = torch.ones((x.shape[0],0)).cuda()
            in_latents = x[:,0:self.latent_size]
        else:
            inputDataDims = 6
            in_x = x[:,-6:]
            if global_feats:
                feats, globalVal = currFeatPL(x[:,-6:].detach())
            elif currFeatPL is not None:
                feats = currFeatPL(x[:,-6:].detach())
            else:
                feats = torch.ones((x.shape[0],0)).cuda()
            in_latents = x[:,self.latent_size:-inputDataDims]
        

        if not self.pos_invariant:
            if self.positionalEncoding>0:
                netIn = torch.cat([in_latents,positional_encoding(in_x,self.positionalEncoding), feats], dim=-1)
            else:
                netIn = torch.cat([in_latents, in_x, feats], dim=-1)
        else:
            netIn = torch.cat([in_latents, feats], dim=-1)
        # import pdb; pdb.set_trace()
        netOut = self.network(netIn)
        netOut = netOut.squeeze()
        if global_feats:
            netOut += globalVal
        
        return netOut
