from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from models import DLinear
class Learnable_decomp(nn.Module):
    """
    Learnable decomposition block
    """

    def __init__(self, seq_len, pred_len, patch_len, l2_fac=0.001):  # fc_dropout=0.0:
        super(Learnable_decomp, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.l2_fac = l2_fac
        assert np.mod(seq_len, patch_len) == 0, "Error: mod(seq_len,patch_len)!=0 in Learnable_decomp"
        assert np.mod(pred_len, patch_len) == 0, "Error: mod(pred_len,patch_len)!=0 in Learnable_decomp"
        self.patch_num = int(self.seq_len / self.patch_len)
        self.kernel_num = max(int(np.floor(np.log(seq_len - 1) / np.log(2))) - 1, 1)
        self.avg_pool_patch = nn.AvgPool1d(patch_len, stride=patch_len, padding=0)
        self.patch_num_pred = int(self.pred_len / self.patch_len)
        # self.fc_trend =nn.Sequential(nn.Dropout(fc_dropout), nn.Linear(self.patch_num, self.patch_num_pred, bias=False))
        self.fc_trend = nn.Linear(self.patch_num, self.patch_num_pred, bias=False)
        self.fc_trend.weight = nn.Parameter(torch.zeros([self.patch_num_pred, self.patch_num]),
                                            requires_grad=True)  # (1/self.patch_num_pred) * torch.ones([self.patch_num_pred, self.patch_num])
        # Weights of modes before softmax
        self.importance_mods = nn.Parameter(torch.zeros(self.kernel_num, dtype=torch.float32), requires_grad=True)
        # self.weight_mods = nn.Parameter(torch.ones(self.kernel_num-1, dtype=torch.float32)/self.kernel_num, requires_grad=True) #no weight for mod 0 for the sum of weights is 1
        self.fn_pads = [nn.ReplicationPad1d(2 ** (i - 1)) for i in range(1, self.kernel_num)]  # no padding for mod 0
        self.avg_pools = [nn.AvgPool1d(2 ** i + 1, stride=1, padding=0) for i in
                          range(1, self.kernel_num)]  # no avg_pool for mod 0

    def forward(self, x):
        # x: [bs, seq_len, c_in]
        x_transpose = x.permute(0, 2, 1)  # x_transpose: [bs, c_in, seq_len]
        t_patched = self.avg_pool_patch(x_transpose)  # t_patched: [bs, c_in, patch_num]
        weight_mods = F.softmax(self.importance_mods, dim=0)
        # trend: [bs, c_in, patch_num]
        trend = t_patched * weight_mods[0]  # (1.0 - self.weight_mods.sum())
        for i_mod in range(0, self.kernel_num - 1):
            t_padded = self.fn_pads[i_mod](t_patched)
            mod_cur = self.avg_pools[i_mod](t_padded)
            trend = trend + mod_cur * weight_mods[i_mod + 1]  # self.weight_mods[i_mod]
        trend_fine = F.interpolate(trend, scale_factor=self.patch_len, mode='linear', align_corners=False)
        season = x_transpose - trend_fine  # season: [bs, c_in, seq_len]
        season = season.permute(0, 2, 1)  # season: [bs, seq_len, c_in]
        trend_pred = self.fc_trend(trend)  # trend_pred: [bs, c_in, patch_num_pred]
        # trend_pred_fine: [bs, channel, pred_len]
        trend_pred_fine = F.interpolate(trend_pred, scale_factor=self.patch_len, mode='linear', align_corners=False)
        trend_pred_fine = trend_pred_fine.permute(0, 2, 1)
        # L2 regularization for trend prediction
        if self.fc_trend.weight.grad != None:
            self.fc_trend.weight.grad.data.add_(self.l2_fac * self.fc_trend.weight.data)
        return season, trend_pred_fine  # season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, channel]


class Multi_decomp(nn.Module):
    """
    multi-scale hybrid decomposition proposed by MICN
    """

    def __init__(self, seq_len, pred_len, kernel_sizes=[17, 49], patch_len=1, l2_fac=0.001):  # fc_dropout=0.0:
        super(Multi_decomp, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.l2_fac = l2_fac
        assert np.mod(seq_len, patch_len) == 0, "Error: mod(seq_len,patch_len)!=0 in Multi_decomp"
        assert np.mod(pred_len, patch_len) == 0, "Error: mod(pred_len,patch_len)!=0 in Multi_decomp"
        for kernel_size in kernel_sizes:
            assert kernel_size > 0 and (
                        kernel_size - 1) % 2 == 0, "Error: kernel_size in Multi_decomp should be positive odd"
        self.patch_num = int(self.seq_len / self.patch_len)
        self.kernel_num = len(kernel_sizes)
        self.avg_pool_patch = nn.AvgPool1d(patch_len, stride=patch_len, padding=0)
        self.patch_num_pred = int(self.pred_len / self.patch_len)
        # self.fc_trend =nn.Sequential(nn.Dropout(fc_dropout), nn.Linear(self.patch_num, self.patch_num_pred, bias=False))
        self.fc_trend = nn.Linear(self.patch_num, self.patch_num_pred, bias=False)
        self.fc_trend.weight = nn.Parameter(
            (1 / self.patch_num_pred) * torch.ones([self.patch_num_pred, self.patch_num]), requires_grad=True)
        self.fn_pads = [nn.ReplicationPad1d(int((kernel_size - 1) / 2)) for kernel_size in kernel_sizes]
        self.avg_pools = [nn.AvgPool1d(kernel_size, stride=1, padding=0) for kernel_size in
                          kernel_sizes]  # no avg_pool for mod 0

    def forward(self, x):
        # x: [bs, seq_len, c_in]
        x_transpose = x.permute(0, 2, 1)  # x_transpose: [bs, c_in, seq_len]
        t_patched = self.avg_pool_patch(x_transpose)  # t_patched: [bs, c_in, patch_num]
        # trend: [bs, c_in, patch_num]
        for i_mod in range(0, self.kernel_num):
            t_padded = self.fn_pads[i_mod](t_patched)
            mod_cur = self.avg_pools[i_mod](t_padded)
            if i_mod == 0:
                trend = mod_cur
            else:
                trend = trend + mod_cur
        trend = trend / self.kernel_num
        trend_fine = F.interpolate(trend, scale_factor=self.patch_len, mode='linear', align_corners=False)
        season = x_transpose - trend_fine  # season: [bs, c_in, seq_len]
        season = season.permute(0, 2, 1)  # season: [bs, seq_len, c_in]
        trend_pred = self.fc_trend(trend)  # trend_pred: [bs, c_in, patch_num_pred]
        # trend_pred_fine: [bs, channel, pred_len]
        trend_pred_fine = F.interpolate(trend_pred, scale_factor=self.patch_len, mode='linear', align_corners=False)
        trend_pred_fine = trend_pred_fine.permute(0, 2, 1)
        # L2 regularization for trend prediction
        if self.fc_trend.weight.grad != None:
            self.fc_trend.weight.grad.data.add_(self.l2_fac * self.fc_trend.weight.data)
        return season, trend_pred_fine  # season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, channel]


class Dlinear_decomp(nn.Module):
    """
    Dlinear decomposition
    """

    def __init__(self, seq_len, pred_len, c_in):  # fc_dropout=0.0:
        super(Dlinear_decomp, self).__init__()

        class CfgDlinear:
            def __init__(self, seq_len, pred_len, enc_in):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.individual = False
                self.enc_in = enc_in

        self.model = DLinear.Model(CfgDlinear(seq_len, pred_len, c_in)).float()

    def forward(self, x):
        # x: [bs, seq_len, c_in]
        season = x
        trend_pred_fine = self.model(x)
        return season, trend_pred_fine  # season: [bs, seq_len, c_in], trend_pred_fine: [bs, pred_len, c_in]
