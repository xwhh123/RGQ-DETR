"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        # Pass router + distill info for auxiliary losses
        if hasattr(self.encoder, '_router_score') and self.encoder._router_score is not None:
            x['router_score'] = self.encoder._router_score
        if hasattr(self.encoder, '_router_target') and self.encoder._router_target is not None:
            x['router_target'] = self.encoder._router_target
        if hasattr(self.encoder, '_distill_loss') and self.encoder._distill_loss is not None:
            x['distill_loss'] = self.encoder._distill_loss

        return x
    
    def deploy(self, ):
        self.eval()
        self.encoder._deploy = True
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
