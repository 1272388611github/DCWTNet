import torch
import torch.nn as nn

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:  #仿射变换参数使得该模块在归一化的同时保留一定的可学习能力，从而适应数据的变化。
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class FrequencyNormalization(nn.Module):
    def __init__(self, eps=1e-5):
        super(FrequencyNormalization, self).__init__()
        self.eps = eps

    def forward(self, x):
        # 进行傅里叶变换，转换到频域
        freq_x = torch.fft.fft(x, dim=1)

        # 计算频域特征的均值和标准差
        mean = torch.mean(freq_x, dim=1, keepdim=True).detach()
        std = torch.sqrt(torch.var(freq_x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

        # 进行归一化
        norm_freq_x = (freq_x - mean) / std

        # 进行逆傅里叶变换，返回到时域
        norm_x = torch.fft.ifft(norm_freq_x, dim=1).real

        return norm_x