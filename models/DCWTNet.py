import torch
import torch.nn as nn
import torch.fft
from layers.Embed import Patch_FFWEmbedding,FlattenHead
from layers.Conv_Blocks import Inception_Block_V1
from layers.RevIN import RevIN
import ptwt
import numpy as np
import math


def Wavelet_for_Period(x, scale=16):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs

def DCT_transform(x):
    N = x.size(1)
    device = x.device

    k = torch.arange(N, device=device).unsqueeze(1)
    n = torch.arange(N, device=device).unsqueeze(0)
    dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k)
    X = torch.matmul(dct_matrix, x)

    second_dim_size = X.shape[1]
    retain_percentage = 0.5
    retain_count = int(second_dim_size * retain_percentage)

    X[:, retain_count:, :] = 0

    X[:, 0, :] *= 1 / math.sqrt(N)
    X[:, 1:, :] *= math.sqrt(2 / N)
    X = X.unsqueeze(2).repeat(1, 1, 8, 1)
    X = X.permute(0, 3, 2, 1)

    return  X

class DCWTBlock(nn.Module):
    def __init__(self, configs):
        super(DCWTBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = configs.wavelet_scale
        #patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.nf = configs.d_model * \
                  int((self.seq_len - self.patch_len) / self.stride + 1)
        self.patch_n = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.projection = nn.Linear(self.patch_n, self.patch_n, bias=True)

        self.mixconv = nn.Sequential(

            Inception_Block_V1(2*configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels),
        )
        self.conv = nn.Sequential(

            Inception_Block_V1( configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        self.scale_conv = nn.Conv2d(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            kernel_size=(8, 1),
            stride=1,
            padding=(0, 0),
            groups=configs.d_model)

    def forward(self, x):

        coeffs = Wavelet_for_Period(x.permute(0, 2, 1), self.scale)[0].permute(1, 2, 0, 3).float()
        dct_x= DCT_transform(x)

        combined_input = torch.cat([dct_x,coeffs], dim=1)
        mixres = self.mixconv(combined_input)
        mixres = self.scale_conv(mixres).squeeze(2).permute(0, 2, 1)
        res = mixres  #224 11 16
        res = res + x

        return self.projection(res.permute(0, 2, 1)).permute(0, 2, 1)  #patch间



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([DCWTBlock(configs)
                                    for _ in range(configs.e_layers)])

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        #patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.dropout = 0
        self.enc_embedding = Patch_FFWEmbedding(
            configs.d_model,
            self.patch_len,
            self.stride,
            8,
            self.dropout,
        )
        self.cross_projection = nn.Linear(configs.c_out, configs.c_out, bias=True)

        self.nf = int(configs.d_model * ((self.seq_len - self.patch_len) / self.stride + 1))
        self.patch_n = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.head = FlattenHead(configs.c_out, self.nf, self.pred_len, 0)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.revin_layer(x_enc, 'norm')
        # embedding
        enc_out ,n_vars = self.enc_embedding(x_enc)  #[bs*features,patch_num,d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[1], enc_out.shape[2])) #[bs,features,patch_num,d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2,1)
        dec_out = self.revin_layer(dec_out, 'denorm')
        dec_out = dec_out

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None


