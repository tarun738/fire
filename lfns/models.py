import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
import lfn.util as util
import lfn.conv_modules as conv_modules
import lfn.custom_layers as custom_layers
import lfn.geometry as geometry
import lfn.hyperlayers as hyperlayers


class LightFieldModel(nn.Module):
    def __init__(self, latent_dim, parameterization='plucker', network='relu',
                 fit_single=False, conditioning='hyper'):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.fit_single = fit_single
        self.parameterization = parameterization
        self.conditioning = conditioning
        self.output_dim = 2
        out_channels = 2

        if self.fit_single or conditioning in ['hyper', 'low_rank']:
            if network == 'relu':
                self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=6,
                                                 in_features=6, out_features=out_channels, outermost_linear=True, norm='layernorm_na')
            elif network == 'siren':
                omega_0 = 30.
                self.phi = custom_layers.Siren(in_features=6, hidden_features=256, hidden_layers=8,
                                               out_features=out_channels, outermost_linear=True, hidden_omega_0=omega_0,
                                               first_omega_0=omega_0)
        elif conditioning == 'concat':
            self.phi = nn.Sequential(
                nn.Linear(6+self.latent_dim, self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                nn.Linear(self.num_hidden_units_phi, 3)
            )

        if not self.fit_single:
            if conditioning=='hyper':
                self.hyper_phi = hyperlayers.HyperNetwork(hyper_in_features=self.latent_dim,
                                                          hyper_hidden_layers=1,
                                                          hyper_hidden_features=self.latent_dim,
                                                          hypo_module=self.phi)
            elif conditioning=='low_rank':
                self.hyper_phi = hyperlayers.LowRankHyperNetwork(hyper_in_features=self.latent_dim,
                                                                 hyper_hidden_layers=1,
                                                                 hyper_hidden_features=512,
                                                                 hypo_module=self.phi,
                                                                 nonlinearity='leaky_relu')
        self.sigm = nn.Sigmoid()
        print(self.phi)
        print(np.sum(np.prod(param.shape) for param in self.phi.parameters()))

    def get_light_field_function(self, z=None):
        if self.fit_single:
            phi = self.phi
        elif self.conditioning in ['hyper', 'low_rank']:
            phi_weights = self.hyper_phi(z)
            phi = lambda x: self.phi(x, params=phi_weights)
        elif self.conditioning == 'concat':
            def phi(x):
                b, n_pix = x.shape[:2]
                z_rep = z.view(b, 1, self.latent_dim).repeat(1, n_pix, 1)
                return self.phi(torch.cat((z_rep, x), dim=-1))
        return phi


    def forward(self, x, getColor=False, inputDataDims=3, addFeats=None, inglobfeats=False):
        x = x.unsqueeze(1)
        
        if self.parameterization == 'plucker':
            light_field_coords = geometry.plucker_embedding(x[...,-6:-3],x[...,-3:])
        else:
            ray_origin = x[...,-6:-3]
            ray_dir = x[...,-3:]

            light_field_coords = torch.cat((ray_origin, ray_dir), dim=-1)

        light_field_coords.requires_grad_(True)

        lf_function = self.get_light_field_function(x[...,:-6])

        lf_out = lf_function(light_field_coords)
        lf_out = lf_out.squeeze()
        depth = lf_out[..., 0]
        alpha = self.sigm(lf_out[..., 1])
        
        return depth, alpha


class LFEncoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', conditioning='hyper'):
        super().__init__(latent_dim, parameterization, conditioning='low_rank')
        self.num_instances = num_instances
        self.encoder = conv_modules.Resnet18(c_dim=latent_dim)

    def get_z(self, input, val=False):
        n_qry = input['query']['uv'].shape[1]
        rgb = util.lin2img(util.flatten_first_two(input['context']['rgb']))
        z = self.encoder(rgb)
        z = z.unsqueeze(1).repeat(1, n_qry, 1)
        z *= 1e-2
        return z
