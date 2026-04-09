from .cbam import ChannelPool, BasicConv
import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class Integration(nn.Module):
    def __init__(self, img_size):
        super(Integration, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.unsample = nn.UpsamplingNearest2d(size=(img_size, img_size))
    def forward(self, x):
        # x_out = torch.mean(x, 1).unsqueeze(1)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        output = torch.sigmoid(x_out)  # broadcasting
        op = self.unsample(output)
        return op

def build_img(model, x1, x2, x3):
    op1 = model(x1)
    op2 = model(x2)
    op3 = model(x3)
    image = torch.cat((op1, op2, op3), dim=1)
    return image

def build_img2(model, x):
    op = model(x)
    return op

class model_fusion(nn.Module):
    def __init__(self, channel_nums):
        super(model_fusion, self).__init__()
        kernel_size = 7
        self.conv = BasicConv(2048, 1024, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.fc = nn.Linear(channel_nums, 2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool, lastfea_return=False):
        if global_pool.size(1) != local_pool.size(1):
            if global_pool.size(1) == 2048:
                global_pool = self.conv(global_pool)
            elif local_pool.size(1) == 2048:
                local_pool = self.conv(local_pool)

        output = torch.cat((global_pool, local_pool), 1)
        res = output.view(output.size(0), -1)
        x = self.fc(res)
        # x = self.Sigmoid(x)
        if lastfea_return:
            return x,res
        else:
            return x
