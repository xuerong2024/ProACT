'Weakly Supervised Lesion Localization With Probabilistic-CAM Pooling, 2020'
'https://github.com/jfhealthcare/Chexpert/tree/master'
from torch import nn

import torch.nn.functional as F
from model_cls.densenet import (densenet121, densenet169, densenet201)

import torch
def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))
class Conv2dNormRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 bias=True, norm_type='Unknown'):
        super(Conv2dNormRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            get_norm(norm_type, out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class CAModule(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    code reference:
    https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    """

    def __init__(self, num_channels, reduc_ratio=2):
        super(CAModule, self).__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio,
                             bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels,
                             bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # attention branch--squeeze operation
        gap_out = feat_map.view(feat_map.size()[0], self.num_channels,
                                -1).mean(dim=2)

        # attention branch--excitation operation
        fc1_out = self.relu(self.fc1(gap_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))

        # attention operation
        fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1, 1)
        feat_map = torch.mul(feat_map, fc2_out)

        return feat_map


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1,
                                               width * height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dims=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class FPAModule(nn.Module):
    """
    Re-implementation of feature pyramid attention (FPA) described in:
    *Li et al., Pyramid Attention Network for Semantic segmentation, Face++2018
    """

    def __init__(self, num_channels, norm_type):
        super(FPAModule, self).__init__()

        # global pooling branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dNormRelu(num_channels, num_channels, kernel_size=1,
                           norm_type=norm_type))

        # middle branch
        self.mid_branch = Conv2dNormRelu(num_channels, num_channels,
                                         kernel_size=1, norm_type=norm_type)

        self.downsample1 = Conv2dNormRelu(num_channels, 1, kernel_size=7,
                                          stride=2, padding=3,
                                          norm_type=norm_type)

        self.downsample2 = Conv2dNormRelu(1, 1, kernel_size=5, stride=2,
                                          padding=2, norm_type=norm_type)

        self.downsample3 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2,
                                          padding=1, norm_type=norm_type)

        self.scale1 = Conv2dNormRelu(1, 1, kernel_size=7, padding=3,
                                     norm_type=norm_type)
        self.scale2 = Conv2dNormRelu(1, 1, kernel_size=5, padding=2,
                                     norm_type=norm_type)
        self.scale3 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1,
                                     norm_type=norm_type)

    def forward(self, feat_map):
        height, width = feat_map.size(2), feat_map.size(3)
        gap_branch = self.gap_branch(feat_map)
        gap_branch = nn.Upsample(size=(height, width), mode='bilinear',
                                 align_corners=False)(gap_branch)

        mid_branch = self.mid_branch(feat_map)

        scale1 = self.downsample1(feat_map)
        scale2 = self.downsample2(scale1)
        scale3 = self.downsample3(scale2)

        scale3 = self.scale3(scale3)
        scale3 = nn.Upsample(size=(height // 4, width // 4), mode='bilinear',
                             align_corners=False)(scale3)
        scale2 = self.scale2(scale2) + scale3
        scale2 = nn.Upsample(size=(height // 2, width // 2), mode='bilinear',
                             align_corners=False)(scale2)
        scale1 = self.scale1(scale1) + scale2
        scale1 = nn.Upsample(size=(height, width), mode='bilinear',
                             align_corners=False)(scale1)

        feat_map = torch.mul(scale1, mid_branch) + gap_branch

        return feat_map


class AttentionMap(nn.Module):

    def __init__(self, cfg, num_channels):
        super(AttentionMap, self).__init__()
        self.cfg = cfg
        self.channel_attention = CAModule(num_channels)
        self.spatial_attention = SAModule(num_channels)
        self.pyramid_attention = FPAModule(num_channels, cfg.norm_type)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map):
        if self.cfg.attention_map == "CAM":
            return self.channel_attention(feat_map)
        elif self.cfg.attention_map == "SAM":
            return self.spatial_attention(feat_map)
        elif self.cfg.attention_map == "FPA":
            return self.pyramid_attention(feat_map)
        elif self.cfg.attention_map == "None":
            return feat_map
        else:
            Exception('Unknown attention type : {}'
                      .format(self.cfg.attention_map))
class PcamPool(nn.Module):
    def __init__(self):
        super(PcamPool, self).__init__()

    def forward(self, feat_map, logit_map):
        assert logit_map is not None

        prob_map = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)

        return feat


class pcam_dense121(nn.Module):
    def __init__(self, num_classes=[1,1]):
        super(pcam_dense121, self).__init__()
        self.num_classes = num_classes
        self.backbone = densenet121(pretrainedpt='/disk3/wjr/workspace/densenet121-a639ec97.pth', norm_type="BatchNorm")
        self.global_pool = PcamPool()
        self.expand = 1

        self._init_classifier()
        self._init_bn()

    def _init_classifier(self):
        for index, num_class in enumerate(self.num_classes):
            setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.num_classes):
            setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            logit_map = classifier(feat_map)
            logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)
            # feat = F.dropout(feat, p=self.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)
        logits=torch.stack(logits).squeeze(-1).permute(1,0)
        return logits

class dense121(nn.Module):
    def __init__(self, num_classes=[1,1]):
        super(dense121, self).__init__()
        self.num_classes = num_classes
        self.backbone = densenet121(pretrainedpt='/disk3/wjr/workspace/densenet121-a639ec97.pth', norm_type="BatchNorm")
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.expand = 1

        self._init_classifier()
        self._init_bn()

    def _init_classifier(self):
        for index, num_class in enumerate(self.num_classes):
            setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.num_classes):
            setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            # # (N, 1, H, W)
            # logit_map = None
            # logit_map = classifier(feat_map)
            # logit_maps.append(logit_map.squeeze())
            # # (N, C, 1, 1)
            # feat = self.global_pool(feat_map, logit_map)
            # # feat = F.dropout(feat, p=self.fc_drop, training=self.training)
            # # (N, num_class, 1, 1)
            feat = self.global_pool(feat_map)
            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)
        logits=torch.stack(logits).squeeze(-1).permute(1,0)
        return logits