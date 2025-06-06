import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print(x.size())  #224 512 8  -> 224 512 8 9
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Inception_Block_V1_parll(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1_parll, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print(x.size())  #224 512 8  -> 224 512 8 9
        x = x.unsqueeze(-1)
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        res = res.squeeze(-1)
        return res

class TCN_Block(nn.Module):
        def __init__(self, in_channels, out_channels, num_kernels=6, kernel_size=3, init_weight=True):
            super(TCN_Block, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_kernels = num_kernels
            self.kernel_size = kernel_size

            self.tcn_layers = nn.ModuleList()
            for i in range(self.num_kernels):
                dilation_size = 3 ** i
                padding = (self.kernel_size - 1) * dilation_size // 2
                self.tcn_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.kernel_size),
                              padding=(0, padding), dilation=(1, dilation_size))  # ,groups=8)
                )

            self.conv1x1 = nn.Conv2d(out_channels * self.num_kernels, out_channels, kernel_size=1)

            if init_weight:
                self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            res_list = []
            for layer in self.tcn_layers:
                res_list.append(layer(x))
            res = torch.cat(res_list, dim=1)
            res = self.conv1x1(res)
            return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
if __name__ == '__main__':
    import torch


    print(torch.cuda.device_count())  # 打印可用的 GPU 数量
    print(torch.cuda.current_device())  # 检查当前设备 ID


