# Taken and modified from https://github.com/sicxu/Deep3DFaceRecon_pytorch

import torch
import torch.nn as nn

import os

from HYDRO.model.resnet import conv1x1, resnet18, resnet50, filter_state_dict

func_dict = {
    'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048)
}


class Encoder3DMM(nn.Module):
    def __init__(self, net_recon='resnet50', use_last_fc=False, init_path=None):
        super(Encoder3DMM, self).__init__()
        self.use_last_fc = use_last_fc
        self.fc_dim = 257
        if net_recon not in func_dict:
            raise NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("loading init net_recon %s from %s" %(net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True),   # id layer
                conv1x1(last_dim, 64, bias=True),   # exp layer
                conv1x1(last_dim, 80, bias=True),   # tex layer
                conv1x1(last_dim, 3, bias=True),    # angle layer
                conv1x1(last_dim, 27, bias=True),   # gamma layer
                conv1x1(last_dim, 2, bias=True),    # tx, ty
                conv1x1(last_dim, 1, bias=True)     # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(torch.flatten(layer(x), 1))
            #x = torch.flatten(torch.cat(output, dim=1), 1)
        return output





