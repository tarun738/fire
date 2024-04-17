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
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.posEncode import positional_encoding

class FeaturePlanes(nn.Module):
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
        self.fm = nn.ParameterList()
        for i in range(0,self.indim-1):
            for j in range(i+1,self.indim):
                self.fm.append(nn.Parameter(torch.randn(1, self.fdim, self.fsize+1, self.fsize+1) * 0.01))
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
        g_h = (( feats[:,:,1:,1:] - feats[:,:,1:,:-1] )**2)
        g_w = (( feats[:,:,1:,1:] - feats[:,:,:-1,1:] )**2)
        loss = g_h+g_w
        if self.tvlsqrt:
            loss = (loss+1e-10)**0.5
        return loss.mean()
class attnCondBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, hidden_dim, num_mlp_layers, act='relu',last_layer_out_dim=None,last_layer_act=True,haveAttn=True, in_cat_dim=0):
        super(attnCondBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.haveAttn = haveAttn
        self.in_cat_dim = in_cat_dim
        if last_layer_out_dim is None:
            self.last_layer_out_dim = self.hidden_dim
        else:
            self.last_layer_out_dim = last_layer_out_dim
        self.last_layer_act = last_layer_act
        self.num_mlp_layers = num_mlp_layers
        self.act = act
        if self.act == 'relu':
            self.activation = nn.ReLU
        if self.haveAttn:
            self.attnLayer = nn.MultiheadAttention(self.embed_dim, num_heads,kdim=self.latent_dim,vdim=self.latent_dim)
        layers = []
        netInDim = self.embed_dim if self.haveAttn else self.embed_dim+self.latent_dim
        layers.append(nn.Linear(netInDim+self.in_cat_dim, self.hidden_dim, bias=True))
        layers.append(self.activation())
        for i in range(self.num_mlp_layers-1):
            if i == self.num_mlp_layers-2:
                out_dim = self.last_layer_out_dim
                actLayer = self.last_layer_act
            else:
                out_dim = self.hidden_dim
                actLayer = True
            layers.append(nn.Linear(self.hidden_dim, out_dim, bias=True))
            if actLayer:
                layers.append(self.activation())
        self.network = nn.Sequential(*layers)
    def forward(self, query, key, value, inCat=None):
        if self.haveAttn:
            netInt,_ = self.attnLayer(query.unsqueeze(0),key.unsqueeze(0),value.unsqueeze(0),need_weights=False)
        else:
            netInt = torch.cat([query,key],dim=-1)
        if self.in_cat_dim >0:
            netInt = torch.cat([netInt.squeeze(),inCat],dim=-1)
        netOut = self.network(netInt.squeeze())
        return netOut
        

class DDFNet(nn.Module):
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
                       haveAttn = False,
                       attentionIn = [0,2],
                       numAttnHeads = 16,
                       featAttn=True,
                       latentAttn=True,
                       splitLat=False,
                       numBlocks=2,
                       output_dim=1):

        super(DDFNet, self).__init__()
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
        if self.fdim ==0:
            self.features_planes = None
            self.numplanes = 0
        else:
            self.features_planes = FeaturePlanes(self.fdim, self.fsize, self.input_dim, self.tvlsqrt, globalFeats=self.global_feats)
            self.numplanes = len(self.features_planes.fm)
        self.network = None
        self.numAttnHeads = numAttnHeads
        self.haveAttn = haveAttn
        self.featAttn = featAttn
        self.latentAttn = latentAttn
        self.output_dim = output_dim

        # concatenate featureplane features, so n* the feature dimension
        self.latent_input_size = self.numplanes*self.fdim + self.latent_size

        # positional encoding, add the input positionally  encoded (or just xyz and/or dir) to the input dimension
        if not self.pos_invariant:
            self.encoded_input_dim = self.input_dim + self.input_dim * 2 * self.positionalEncoding # if self.input_dim==3 else 4
        else:
            self.encoded_input_dim = 0
        
        self.sdf_input_dim = self.encoded_input_dim + self.latent_input_size

        self.keySize = self.latent_input_size
        self.querySize = self.encoded_input_dim
        in_cat_dim = 0
        if (not self.featAttn) and haveAttn:
            self.keySize = self.latent_input_size - self.numplanes*self.fdim
            in_cat_dim = self.numplanes*self.fdim
        elif (not self.latentAttn) and haveAttn:
            self.keySize = self.latent_input_size - self.latent_size
            in_cat_dim = self.latent_size
        layers = []
        layers.append(attnCondBlock(self.encoded_input_dim, 7, self.keySize, self.hidden_dim, num_mlp_layers=3,last_layer_out_dim=self.output_dim if numBlocks== 1 else self.hidden_dim-self.encoded_input_dim,haveAttn=self.haveAttn,in_cat_dim=in_cat_dim))
        if numBlocks>1:
            layers.append(attnCondBlock(self.hidden_dim, self.numAttnHeads, self.keySize, self.hidden_dim, num_mlp_layers=3,last_layer_out_dim=self.output_dim if numBlocks== 2 else self.hidden_dim-self.encoded_input_dim,last_layer_act=False,haveAttn=self.haveAttn,in_cat_dim=in_cat_dim))
        if numBlocks>2:
            layers.append(attnCondBlock(self.hidden_dim, self.numAttnHeads, self.keySize, self.hidden_dim, num_mlp_layers=3,last_layer_out_dim=self.output_dim,last_layer_act=False,haveAttn=self.haveAttn,in_cat_dim=in_cat_dim))
        self.network = nn.Sequential(*layers)
        # import pdb; pdb.set_trace()
        self.sigm = nn.Sigmoid()
    def getFeatTVL(self):
        if self.features_planes is not None:
            return self.features_planes.getTVL()
        else:
            return 0
        
    def forward(self,x, inputDataDims=3, addFeats=None, inglobfeats=False):
        # import pdb; pdb.set_trace()
        global_feats = self.global_feats if not inglobfeats else inglobfeats
        if addFeats is None:
            currFeatPlanes = self.features_planes
        else:
            currFeatPlanes = addFeats
        if self.input_dim == 3:
            inputDataDims = 3
            in_latent = x[:,:-inputDataDims]
            if self.splitLat:
                in_latent = x[:,0:self.latent_size]
        else:
            inputDataDims = 6
            in_latent = x[:,:-inputDataDims]
            
            if self.splitLat:
                in_latent = x[:,self.latent_size:-inputDataDims]
        if not self.splitLat:
            in_latent = x[:,:-inputDataDims]
        in_x = x[:,-inputDataDims:]
        if global_feats:
            feats, globalVal = currFeatPlanes(x[:,-inputDataDims:].detach())
        else:
            if currFeatPlanes is not None:
                feats = currFeatPlanes(x[:,-inputDataDims:].detach())
            else:
                feats = torch.ones((x.shape[0],0)).cuda()

        if not self.pos_invariant:
            if self.positionalEncoding>0:
                encoded_in = positional_encoding(in_x,self.positionalEncoding)
            else:
                encoded_in = in_x
        else:
            netIn = feats
        if not self.latentAttn:
            latent_cond = feats
            inCat = in_latent
        elif not self.featAttn:
            latent_cond = in_latent 
            inCat = feats
        else:
            latent_cond = torch.cat([in_latent,feats],dim=-1)
            inCat = None
        netOut = self.network[0](encoded_in,latent_cond,latent_cond,inCat=inCat)
        if len(self.network)>1:
            netOut = self.network[1](torch.cat([netOut,encoded_in],dim=-1),latent_cond,latent_cond,inCat=inCat)
        if len(self.network)>2:
            netOut = self.network[2](torch.cat([netOut,encoded_in],dim=-1),latent_cond,latent_cond,inCat=inCat)
        
        netOut = netOut.squeeze()
        if self.output_dim>1:
            hitmissOut = self.sigm(netOut[:,1:])
            netOut =  netOut[:,0]
        if global_feats:
            netOut += globalVal
        
        if self.output_dim>1:
            return netOut,hitmissOut
        else:
            return netOut