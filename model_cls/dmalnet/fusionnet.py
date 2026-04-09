from torch import nn
from model_cls.dmalnet.resnet_cbam import ResidualNet
from model_cls.dmalnet.resnet_active import arlnet50
from model_cls.dmalnet.mymodels import *
import os
class fusion_net(nn.Module):
    def __init__(self, pretrained_pth=None):
        super(fusion_net, self).__init__()
        self.first_model = arlnet50(pretrained=False, num_classes=2)
        self.second_model = ResidualNet(network_type="ImageNet", depth=50, num_classes=2, att_type='CBAM')
        self.fusion_model = model_fusion(4096)
        self.temp_model = Integration(256)
        if pretrained_pth!=None:
            name_model = 'model_first.pt'
            path = os.path.join(pretrained_pth, name_model)
            first_model_pt = torch.load(path)
            msg = self.first_model.load_state_dict(first_model_pt, strict=True)
            print(msg)
            name_model = 'model_temp.pt'
            name_model2 = 'model_second.pt'
            path = os.path.join(pretrained_pth, name_model)
            path2 = os.path.join(pretrained_pth, name_model2)
            temp_model_pt = torch.load(path)
            msg = self.temp_model.load_state_dict(temp_model_pt, strict=True)
            print(msg)
            second_model_pt = torch.load(path2)
            msg = self.second_model.load_state_dict(second_model_pt, strict=True)
            print(msg)
            name_model = 'model_fusion.pt'
            path = os.path.join(pretrained_pth, name_model)
            fusion_model_pt = torch.load(path)
            msg = self.fusion_model.load_state_dict(fusion_model_pt, strict=True)
            print(msg)
    def forward(self, x):
        _, out_pool, op1, op2, op3, op4 = self.first_model(x)
        # new_inputs = build_img2(model_temp, op1)
        new_inputs = build_img(self.temp_model, op1, op2, op3)
        # new_inputs = build_img(model_temp, op1, op2, op4)
        new_inputs = new_inputs.cuda()
        _, out_pool2, _, _, _, _ = self.second_model(new_inputs)
        output3 = self.fusion_model(out_pool, out_pool2)
        # print(output3.shape)
        return output3

import time
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
if __name__ == "__main__":

    model = fusion_net().cuda()
    input = torch.randn((1, 3, 224, 224)).cuda()
    timer = Timer()
    timer.tic()
    with torch.no_grad():
        for i in range(100):
            torch.cuda.empty_cache()
            timer.tic()
            y = model(input)
            # print(y.shape)
            timer.toc()

    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))

