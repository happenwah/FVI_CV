import torch.nn as nn
import torch.nn.functional as F
import torch
from .model.densenet import FCDenseNet_103

class Q_FCDenseNet103_FVI(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4,5,7,10,12),
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15,
                 growth_rate=16, out_chans_first_conv=48, L=20, 
                 out_chans=None, p=0., diag=True):
        super().__init__()
        self.L = L
        self.out_chans = out_chans
        self.diag = diag
        if out_chans is None:
            if diag:
                out_chans_last = 1 + L + 1 + 1
            else:
                out_chans_last = 1 + L + 1
        else:
            if diag:
                out_chans_last = out_chans + L * out_chans + out_chans + out_chans
            else:
                out_chans_last = out_chans + L * out_chans + out_chans
        self.model = FCDenseNet_103(in_channels, down_blocks, up_blocks,
                                    bottleneck_layers, growth_rate, out_chans_first_conv, 
                                    out_chans_last=out_chans_last, p=p)

    def forward(self, x):
        out = self.model.forward(x)
        if self.out_chans is None:
            C = 1
        else:
            C = self.out_chans
        out_mean = out[:,:C,:,:]
        out_cov = out[:,C:(C + C*self.L),:,:]
        if self.diag:
                out_cov_diag = out[:,(C + C*self.L):(C + C + C*self.L),:,:]
                out_logvar_aleatoric = out[:,(C + C + C*self.L):(C + C + C + C*self.L),:,:]
                return out_mean, out_cov, out_cov_diag, out_logvar_aleatoric
        else:
                out_logvar_aleatoric = out[:,(C + C*self.L):,:,:]
                return out_mean, out_cov, out_logvar_aleatoric

class Q_FCDenseNet103_MCDropout(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4,5,7,10,12),
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15,
                 growth_rate=16, out_chans_first_conv=48, out_chans=1, p=0.2, use_bn=True):

        super().__init__()
        self.out_chans = out_chans
        self.model = FCDenseNet_103(in_channels, down_blocks, up_blocks,
                                    bottleneck_layers, growth_rate, out_chans_first_conv, 
                                    out_chans_last=2*out_chans, p=p, use_bn=use_bn)
                                                                      
    def forward(self, x):
        out = self.model(x)
        out_mean = out[:,:self.out_chans,:,:]
        out_logvar_aleatoric = out[:,self.out_chans:,:,:]
        return out_mean, out_logvar_aleatoric

