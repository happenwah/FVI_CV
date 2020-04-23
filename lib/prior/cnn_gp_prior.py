import torch.nn as nn
import torch.nn.functional as F
from cnn_gp import Sequential, Conv2d, ReLU
from cnn_gp.kernels import ConvKP

class TransposeConv2d(Conv2d):
    def __init__(self, kernel_size, new_size, padding, var_weight, 
                var_bias, stride=1, dilation=1, in_channel_multiplier=1, out_channel_multiplier=1):
        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation,
                        var_weight=var_weight, var_bias=var_bias, in_channel_multiplier=in_channel_multiplier,
                        out_channel_multiplier=out_channel_multiplier)
        self.new_size = new_size

    def propagate(self, kp):
        kp = ConvKP(kp)
        def f(patch):
            patch = F.interpolate(patch, self.new_size, mode="bilinear", align_corners=False)
            return (F.conv2d(patch, self.kernel, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)
                    + self.var_bias)
        return ConvKP(kp.same, kp.diag, f(kp.xy), f(kp.xx), f(kp.yy))

class CNN_GP_prior:
    def __init__(self, out_size, num_channels_output=1):
        self.C = num_channels_output
        var_bias = 0.08
        var_weight = 0.20
        C_3 = 3 ** 2 
        self.model = Sequential(
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            Conv2d(kernel_size=3, padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            TransposeConv2d(kernel_size=3, new_size=(int(0.2*out_size[0]), int(0.2*out_size[1])),
            padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            TransposeConv2d(kernel_size=3, new_size=(int(0.4*out_size[0]), int(0.4*out_size[1])),
            padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            TransposeConv2d(kernel_size=3, new_size=(int(0.6*out_size[0]), int(0.6*out_size[1])),
            padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            TransposeConv2d(kernel_size=3, new_size=(int(0.8*out_size[0]), int(0.8*out_size[1])),
            padding=0, var_weight=C_3*var_weight, var_bias=var_bias),
            ReLU(),
            TransposeConv2d(kernel_size=1, new_size=out_size,
            padding=0, var_weight=var_weight, var_bias=var_bias),
                           )
        self.model.cuda()       
    def compute_K(self, X):
        assert len(X.size()) == 4
        N = X.size(0)
        K = self.model(X, X, same=True)
        K = K.contiguous().view(N, N, -1).permute(0, 2, 1)
        if self.C > 1:
            K = K.repeat(1, self.C, 1)
        return K
