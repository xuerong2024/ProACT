# from mmpretrain.models.backbones.resnet import *
# # Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

eps = 1.0e-5


class BasicBlock(BaseModule):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        drop_path_rate (float or list): stochastic depth rate.
            Default: 0.
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks

        assert len(drop_path_rate
                   ) == num_blocks, 'Please check the length of drop_path_rate'

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=drop_path_rate[0],
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate[i],
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

class resnet50_our_2_5cls(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(resnet50_our_2_5cls, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential()
        self.head.fc  = nn.Linear(2048, 1000)
        self.head.detailed_5_fc = nn.Linear(2048, 5)
        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(resnet50_our_2_5cls, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
    def fea_diff_forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # x_lr = self.conv1(x_lr)
            # x_lr = self.norm1(x_lr)
            # x_lr = self.relu(x_lr)
            # x = x - x_lr

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.fea_diff_forward(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        _5clsout = self.head.detailed_5_fc(_fea)
        return _clsout, _5clsout, x_diff_fea
        # return _clsout, x_diff_fea
        # return _clsout

    def train(self, mode=True):
        super(resnet50_our_2_5cls, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1
import torch.nn.functional as F
class ResNet(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential()
        self.head.fc  = nn.Linear(2048, 1000)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
    def fea_diff_forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # x_lr = self.conv1(x_lr)
            # x_lr = self.norm1(x_lr)
            # x_lr = self.relu(x_lr)
            # x = x - x_lr

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.fea_diff_forward(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1
class ResNet_tree(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet_tree, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential()
        self.head.fc  = nn.Linear(2048, 1000)
        self.head.detailed_sick_fc = nn.Linear(2048, 3)
        self.head.detailed_health_fc = nn.Linear(2048, 2)
        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet_tree, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
    def fea_diff_forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # x_lr = self.conv1(x_lr)
            # x_lr = self.norm1(x_lr)
            # x_lr = self.relu(x_lr)
            # x = x - x_lr

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.fea_diff_forward(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        weights = self.head.fc.weight
        fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # fea_sick = nn.ReLU()(x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        # fea_health = nn.ReLU()(x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
        _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
        _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
        _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
        return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea
        # return _clsout, x_diff_fea
        # return _clsout

    def train(self, mode=True):
        super(ResNet_tree, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1
class ResNet_hf(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet_hf, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential()
        self.head.fc  = nn.Linear(2048, 1000)
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
    def fea_diff_forward(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # x_lr = self.conv1(x_lr)
            # x_lr = self.norm1(x_lr)
            # x_lr = self.relu(x_lr)
            # x = x - x_lr

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.fea_diff_forward(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout

    def train(self, mode=True):
        super(ResNet_hf, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1
class ResNet_hf_tree_attn(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet_hf_tree_attn, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential()
        self.head.fc  = nn.Linear(2048, 1000)
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1 * 32 ** 2, kernel_size=1,
                      stride=1,
                      groups=1, bias=True),
            # nn.ReLU(),
            nn.PixelShuffle(32),
            nn.Sigmoid()
        )
        self.head.detailed_sick_fc = nn.Linear(2048, 3)
        self.head.detailed_health_fc = nn.Linear(2048, 2)
        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet_hf_tree_attn, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
    def fea_diff_forward(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # x_lr = self.conv1(x_lr)
            # x_lr = self.norm1(x_lr)
            # x_lr = self.relu(x_lr)
            # x = x - x_lr

        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
    def forward(self, x, return_attn=False):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.fea_diff_forward(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        attn = self.final_up(x_diff_fea)
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        weights = self.head.fc.weight
        fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
        _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
        _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
        _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
        if return_attn:
            return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea, attn
        else:
            return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea

        # return _clsout

    def train(self, mode=True):
        super(ResNet_hf_tree_attn, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1
class ResNet_seg(nn.Module):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmpretrain.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet_seg, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.init_cfg=init_cfg
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.identify=nn.Identity()
        self.neck=nn.AdaptiveAvgPool2d(1)

        self.segnet = nn.Sequential(nn.Conv2d(2048, 2048//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(2048 // 8, 2048 // 32,kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(2048 // 32, 1, kernel_size=1),
                                    nn.Sigmoid()
                                    )
        self.fusenet = nn.Sequential(nn.Conv2d(1, 2048 // 32, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(2048 // 32, 2048, kernel_size=3, padding=1),
                                    )
        self.classifier = nn.Linear(2048, 2)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x_seg = self.segnet(x)
        x_fuse = x + self.fusenet(x_seg)
        x_fuse = self.relu(x_fuse)
        out = self.neck(x_fuse).flatten(start_dim=1, end_dim=-1)
        _clsout = self.classifier(out)
        return _clsout, x_seg

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer id to set the different learning rates for ResNet.

        ResNet stages:
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1

def resnet50(pretrained=False, drop_path_rate=0.0):
    model = ResNet(depth=50, drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def resnet50_tree(pretrained=False, drop_path_rate=0.0):
    model = ResNet_tree(depth=50, drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def resnet50_2_5cls(pretrained=False, drop_path_rate=0.):
    model = resnet50_our_2_5cls(depth=50, drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def resnet50_hf(pretrained=False):
    model = ResNet_hf(depth=50)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def resnet50_hf_tree_attn(pretrained=False):
    model = ResNet_hf_tree_attn(depth=50)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

if __name__ == '__main__':
    model=resnet50(pretrained=True)
    aa=model.eval()
    aa=torch.rand(1,3,224,224)
    y=model(aa)
    b=1
