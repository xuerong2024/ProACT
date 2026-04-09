import numpy as np
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


#Template_generator
def Template_generator(H, L, radi, ways):
    stride = H/L
    #radi = H/4
    delete = []
    Temp = []
    for i in range (L+1):
        delete.append(i*stride)

    X = np.linspace(0, H-1, H)
    Y = np.linspace(0, H-1, H)

    for n in range(L+1):
        for m in range(L+1):
            if ways == 'L1':
                Z = np.array([
                    [
                        (radi - np.abs((i-stride*n)) - np.abs((j-stride*m)))/radi
                        for j in Y
                    ]
                    for i in X
                ])
                for i in range(H):
                    for j in range(H):
                        if np.abs(i-stride*n) + np.abs(j-stride*m) >radi:
                            Z[i,j] = 0

            else:
                Z = np.array([
                    [
                        (radi**2 - ((i-stride*n) ** 2 + (j-stride*m) ** 2)) ** 0.5 / radi
                        for j in Y
                    ]
                    for i in X
                ])
                for i in range(H):
                    for j in range(H):
                        if (i-stride*n)**2 + (j-stride*m)**2 >radi**2:
                            Z[i,j] = 0
            # Temp.append(dataset_normalized(Z))
            Temp.append(Z)
    return Temp



def max_pool_3x3(x):
    return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

class PKA2_Net(nn.Module):
    def __init__(self, n_class):
        super(PKA2_Net, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1_1 = self._make_layer(64, 3)
        # self.se_block1_1 = SEBS(56, 64)
        self.conv1_1x1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.res_block1_2_down = res_block_down(64, 128)
        # self.se_block1_2 = SEBS(28, 128)
        self.res_block1_2 = self._make_layer(128, 3)
        self.conv1_2x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.res_block1_3_down = res_block_down(128, 256)
        self.res_block1_3 = self._make_layer(256, 5)
        # self.se_block1_3 = SEBS(14, 256)
        self.conv1_3x1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.res_block1_4_down = res_block_down(256, 512)
        self.res_block1_4 = self._make_layer(512, 2)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, n_class)
        self.T_56 = Template_generator(128, 8, 128 / 8, 'l1')
        self.T_56 = torch.from_numpy((np.array(self.T_56)).astype(np.float16)).cuda()
        self.T_28 = Template_generator(64, 8, 64 / 8, 'l1')
        self.T_28 = torch.from_numpy((np.array(self.T_28)).astype(np.float16)).cuda()
        self.T_14 = Template_generator(32, 8, 32 / 8, 'l1')
        self.T_14 = torch.from_numpy((np.array(self.T_14)).astype(np.float16)).cuda()

    def _make_layer(self, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(ResBlock(planes, planes))
        return nn.Sequential(*layers)



    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.pool1_1(out)

        out = self.res_block1_1(out)
        out = SEBS(self.T_56, out, 128, out.shape[0], 64)
        # out = self.se_block1_1(out)
        out = self.conv1_1x1(out)

        out = self.res_block1_2_down(out)
        out = self.res_block1_2(out)
        out = SEBS(self.T_28, out, 64, out.shape[0], 128)
        # out = self.se_block1_2(out)
        out = self.conv1_2x1(out)

        out = self.res_block1_3_down(out)
        out = self.res_block1_3(out)
        out = SEBS(self.T_14, out, 32, out.shape[0], 256)
        # out = self.se_block1_3(out)
        out = self.conv1_3x1(out)

        out = self.res_block1_4_down(out)
        out = self.res_block1_4(out)

        out = self.gap1(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, 4 * out_planes, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(4 * out_planes, out_planes, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class res_block_down(nn.Module):
    def __init__(self, in_channels, kernel_sum, scale=0.1):
        super(res_block_down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, kernel_sum, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(kernel_sum)
        self.conv2 = nn.Conv2d(kernel_sum, 4 * kernel_sum, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(4 * kernel_sum, kernel_sum, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out


def SEBS(Tem, input0, size, batch, channel):
    XHCH = 3
    input0=input0.permute(0,2,3,1)
    input = input0.to(torch.float16)
    in_input = input
    # cn_layer = torch.zeros((batch, size, size, 1), dtype=torch.float16)
    Tem = Tem.reshape(1, size, size, -1, 1)
    input = input.reshape(batch, size, size, 1, -1)
    in_layer_mul = Tem * input
    Tem1 = 2 * Tem / 2
    Tem1[Tem1 > 0] = 1
    tem1_add = torch.sum(Tem1, dim=[1, 2])
    in_layer_GAP = torch.mean(in_layer_mul, dim=[1, 2]) / tem1_add
    GAP_ones = torch.ones_like(in_layer_GAP, dtype=torch.float16)
    GAP_zeros = torch.zeros_like(in_layer_GAP, dtype=torch.float16)
    in_layer_GAP_max = torch.where(in_layer_GAP < torch.unsqueeze(torch.max(in_layer_GAP, dim=-2).values, -2), GAP_zeros, GAP_ones)
    in_layer_GAP_max = in_layer_GAP_max.reshape(batch, 1, 1, -1, 1, channel)
    in_layer_all = torch.unsqueeze(in_layer_mul, -2)
    in_layer_GAP_kmax0 = in_layer_GAP
    atten = in_layer_GAP_max

    for k in range(XHCH - 1):
        in_layer_GAP_kmax0 = torch.where(in_layer_GAP_kmax0 < torch.unsqueeze(torch.max(in_layer_GAP_kmax0, dim=-2).values, -2), in_layer_GAP_kmax0, GAP_zeros)
        in_layer_GAP_kmax = torch.where(in_layer_GAP_kmax0 < torch.unsqueeze(torch.max(in_layer_GAP_kmax0, dim=-2).values, -2), GAP_zeros, GAP_ones)
        in_layer_GAP_kmax = in_layer_GAP_kmax.reshape(batch, 1, 1, -1, 1, channel)
        atten = torch.cat([atten, in_layer_GAP_kmax], -2)

    current_featuremap_kmax = atten * in_layer_all
    current_featuremap_kmax = torch.sum(current_featuremap_kmax, dim=-3)
    current_featuremap_kmax = current_featuremap_kmax.reshape(batch, size, size, -1)
    cn_layer = torch.cat([in_input, current_featuremap_kmax], -1)
    cn_layer = cn_layer.permute(0, 3,1,2)
    return cn_layer.to(torch.float32)


if __name__ == "__main__":
    model = PKA2_Net(n_class=2).cuda()
    input = torch.ones((3, 3, 224, 224)).cuda()
    y = model(input)
    print(y.shape)