import torch.nn
from mmpretrain.models.backbones.convnext import *
from model_cls.image_filter import *
import math
from torchvision.ops import roi_align


class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN

    """

    def __init__(self, output_size, sampling_ratio=-1):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None

    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return

        possible_scales = []
        for s1, s2 in zip(feature_shape, image_shape):
            scale = 2 ** int(math.log2(s1 / s2))
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        self.spatial_scale = possible_scales[0]

    def __call__(self, feature, proposal, image_shape):
        """
        Arguments:
            feature (Tensor[N, C, H, W])
            proposal (Tensor[K, 4])
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])

        """
        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)

        self.setup_scale(feature.shape[-2:], image_shape)
        return roi_align(feature.to(roi), roi, self.spatial_scale, self.output_size[0], self.output_size[1],
                         self.sampling_ratio)
class ConvNeXt_our_pixelshuffle(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1 * 32 ** 2, kernel_size=1,
                      stride=1,
                      groups=1, bias=True),
            # nn.ReLU(),
            nn.PixelShuffle(32),
            nn.Sigmoid()
        )

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, return_attn=False):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea

        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        if return_attn:
            attn = self.final_up(x_diff_fea)
            return _clsout, x_diff_fea, attn
        else:
            return _clsout, x_diff_fea
        # return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_pixelshuffle, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        _fea=x_fea[1]
        # _fea = self.neck(x_fea[0]).flatten(start_dim=1, end_dim=-1)
        # print(_fea.shape)
        _clsout = self.head.fc(_fea)
        # return _clsout
        return _clsout, _fea
        # return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our, self).train(mode)
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters()))
        self._freeze_stages()
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
from model_cls.attention_modules import Block, CrossAttention, CrossAttentionBlock, trunc_normal_
class ConvNeXt_our_pooling(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.cross_attention_block = CrossAttentionBlock(
            dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, 768))
        trunc_normal_(self.query_tokens, std=0.02)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                x=x.flatten(start_dim=2, end_dim=-1).permute(0,2,1)
                q = self.query_tokens.repeat(len(x), 1, 1)
                gap = self.cross_attention_block(q, x).squeeze(1)
                outs.append(gap)
                # gap = x.mean([-2, -1], keepdim=True)
                # outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        _fea=x_fea[1]
        # _fea = self.neck(x_fea[0]).flatten(start_dim=1, end_dim=-1)
        # print(_fea.shape)
        _clsout = self.head.fc(_fea)
        return _clsout
        # return _clsout, _fea
        # return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_pooling, self).train(mode)
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters()))
        self._freeze_stages()
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_prototype(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        # 2. 区域自适应模块 (关键!)
        self.zone_embed = nn.Embedding(6, 768)  # 可学习的位置嵌入
        self.zone_adapter = nn.Sequential(  # 轻量调制网络
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Linear(64, 768 * 2)  # 输出gamma和beta
        )
        self.register_buffer("cache_keys", torch.randn(5, 768))
        self.register_buffer("cache_values", torch.eye(5))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        _fea=x_fea[1]
        # _fea = self.neck(x_fea[0]).flatten(start_dim=1, end_dim=-1)
        # print(_fea.shape)
        _clsout = self.head.fc(_fea)
        # return _clsout
        return _clsout, _fea
        # return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_prototype, self).train(mode)
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters()))
        self._freeze_stages()
        # print('Trainable parameter number: ', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class FPN(nn.Module):
    def __init__(self, in_channels=[192, 384, 768], out_channel=256):
        super().__init__()
        # 1x1卷积调整各阶段通道数至out_channel
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channel, 1) for c in in_channels
        ])
        # 3x3卷积消除上采样伪影
        self.smooth_convs = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        #     nn.ModuleList([
        #     nn.Conv2d(out_channel, out_channel, 3, padding=1) for _ in range(4)
        # ])

    def forward(self, feats):
        # feats: {"C1": [B,96,H/4,W/4], "C2":[B,192,H/8,W/8], "C3":[B,384,H/16,W/16], "C4":[B,768,H/32,W/32]}
        C1, C2, C3, C4 = feats[0], feats[1], feats[2], feats[3]

        # 自顶向下路径
        P4 = self.lateral_convs[2](C4)  # [B,256,H/32,W/32]
        P3 = self.lateral_convs[1](C3) + F.interpolate(P4, scale_factor=2)  # [B,256,H/16,W/16]
        P2 = self.lateral_convs[0](C2) + F.interpolate(P3, scale_factor=2)  # [B,256,H/8,W/8]

        P2 = self.smooth_convs(P2)

        return P2  # 输出融合后的多尺度特征
class LocalLungClassifier(nn.Module):
    def __init__(self, fpn_channel=256, num_classes=5):  # 密集度0-3级
        super().__init__()
        self.num_classes=num_classes
        # 2. 分类头：全局池化 + 全连接
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # 全局池化到1×1
            nn.Flatten(),
            nn.Linear(fpn_channel*2*2, num_classes)
        )

    def forward(self, fpn_feats, lung_rois, spatial_scale=1.0/8):
        """
        fpn_feats: [B, C, H, W]  # FPN输出的特征图（如P2层，对应原始图像1/8缩放）
        lung_rois: [B, 6, 4]    # 6个肺区是相对于原图的原始绝对坐标，例如[256, 0, 512, 512]
        spatial_scale: 原始图像 → FPN特征图的缩放因子（如P2层为1/8）
        """
        B, C, H, W = fpn_feats.shape
        num_rois = 6  # 固定6个肺区

        # ROIAlign池化：输出 [B×6, C, pool_size, pool_size]
        # 假设 rois 是相对于原图的原始坐标
        pooled = roi_align(
            fpn_feats,
            lung_rois,
            output_size=(8, 8),
            spatial_scale=spatial_scale  # 需与坐标转换一致
        )

        # 特征降维与分类
        logits = self.classifier(pooled)  # [B×6, num_classes]
        # 重塑为 [B, 6, num_classes]
        return logits.view(B, num_rois, self.num_classes)
class ConvNeXt_fpn_local_global(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in range(4):
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.fpn = FPN(in_channels=[192, 384, 768], out_channel=256)
        self.local_classifier = LocalLungClassifier(fpn_channel=256, num_classes=5)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in range(4):
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        return tuple(outs)
    def forward(self, x, rois=None):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[-1]
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        if rois!=None:
            local_feat = self.fpn(x_fea)  # {"P1": ..., "P4": ...}
            # 3. 局部六肺区分支
            local_logits = self.local_classifier(local_feat, rois)  # [B,6,4]
            return _clsout, local_logits, x_diff_fea
        else:
            return _clsout
            # return _clsout, x_diff_fea

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_fpn_local_global, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class LocalGlobalLungClassifier(nn.Module):
    def __init__(self, fpn_channel=256, num_classes=5):  # 密集度0-3级
        super().__init__()
        self.num_classes=num_classes
        # 2. 分类头：全局池化 + 全连接
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # 全局池化到1×1
            nn.Flatten(),
            nn.Linear(fpn_channel*2*2, num_classes)
        )

        self.globalclassifier = nn.Sequential(
            nn.Conv2d(fpn_channel, 64, kernel_size=1),  # 1x1 卷积降维
            nn.AdaptiveAvgPool2d((6, 6)),  # 全局池化到1×1
            nn.Flatten(),
            nn.Linear(64*6*6, 2)
        )

    def forward(self, fpn_feats, lung_rois=None, spatial_scale=1.0/8):
        """
        fpn_feats: [B, C, H, W]  # FPN输出的特征图（如P2层，对应原始图像1/8缩放）
        lung_rois: [B, 6, 4]    # 6个肺区是相对于原图的原始绝对坐标，例如[256, 0, 512, 512]
        spatial_scale: 原始图像 → FPN特征图的缩放因子（如P2层为1/8）
        """
        if lung_rois==None:
            # 特征降维与分类
            logits = self.globalclassifier(fpn_feats)  # [B×6, num_classes]
            # 重塑为 [B, 6, num_classes]
            return logits
        else:
            B, C, H, W = fpn_feats.shape
            num_rois = 6  # 固定6个肺区

            # ROIAlign池化：输出 [B×6, C, pool_size, pool_size]
            # 假设 rois 是相对于原图的原始坐标
            pooled = roi_align(
                fpn_feats,
                lung_rois,
                output_size=(8, 8),
                spatial_scale=spatial_scale  # 需与坐标转换一致
            )
            local_logits = self.local_classifier(pooled)
            # 特征降维与分类
            logits = self.globalclassifier(fpn_feats)  # [B×6, num_classes]
            # 重塑为 [B, 6, num_classes]
            return local_logits.view(B, num_rois, self.num_classes), logits

class ConvNeXt_fpn_fine_local_global(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in range(4):
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.fpn = FPN(in_channels=[192, 384, 768], out_channel=256)
        self.local_global_classifier = LocalGlobalLungClassifier(fpn_channel=256, num_classes=5)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in range(4):
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        return tuple(outs)
    def forward(self, x, rois=None):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[-1]
        # _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        # _clsout = self.head.fc(_fea)
        if rois!=None:
            local_feat = self.fpn(x_fea)  # {"P1": ..., "P4": ...}
            # 3. 局部六肺区分支
            local_logits, _clsout = self.local_global_classifier(local_feat, rois)  # [B,6,4]
            return _clsout, local_logits, x_diff_fea
        else:
            local_feat = self.fpn(x_fea)  # {"P1": ..., "P4": ...}
            # 3. 局部六肺区分支
            _clsout = self.local_global_classifier(local_feat)  # [B,6,4]
            # return _clsout, x_diff_fea
            return _clsout

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_fpn_fine_local_global, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

class ConvNeXt_fpn_fine_global(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in range(4):
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        # self.head = nn.Sequential()
        # self.head.fc = nn.Linear(768, 1000)
        # self.neck = nn.AdaptiveAvgPool2d(1)
        self.fpn = FPN(in_channels=[192, 384, 768], out_channel=256)
        self.globalclassifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),  # 1x1 卷积降维
            nn.AdaptiveAvgPool2d((6, 6)),  # 全局池化到1×1
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 2)
        )
        # self.local_global_classifier = LocalGlobalLungClassifier(fpn_channel=256, num_classes=5)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in range(4):
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        return tuple(outs)
    def forward(self, x, rois=None):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[-1]
        local_feat = self.fpn(x_fea)  # {"P1": ..., "P4": ...}
        # 3. 局部六肺区分支
        _clsout = self.globalclassifier(local_feat)  # [B,6,4]
        # return _clsout, x_diff_fea
        return _clsout



    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_fpn_fine_global, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class FPN_x16(nn.Module):
    def __init__(self, in_channels=[384, 768], out_channel=256):
        super().__init__()
        # 1x1卷积调整各阶段通道数至out_channel
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channel, 1) for c in in_channels
        ])
        # 3x3卷积消除上采样伪影
        self.smooth_convs = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        #     nn.ModuleList([
        #     nn.Conv2d(out_channel, out_channel, 3, padding=1) for _ in range(4)
        # ])

    def forward(self, feats):
        # feats: {"C1": [B,96,H/4,W/4], "C2":[B,192,H/8,W/8], "C3":[B,384,H/16,W/16], "C4":[B,768,H/32,W/32]}
        C1, C2, C3, C4 = feats[0], feats[1], feats[2], feats[3]

        # 自顶向下路径
        P4 = self.lateral_convs[1](C4)  # [B,256,H/32,W/32]
        P3 = self.lateral_convs[0](C3) + F.interpolate(P4, scale_factor=2)  # [B,256,H/16,W/16]
        # P2 = self.lateral_convs[0](C2) + F.interpolate(P3, scale_factor=2)  # [B,256,H/8,W/8]

        P2 = self.smooth_convs(P3)

        return P2  # 输出融合后的多尺度特征
class ConvNeXt_fpn_fine_x16_global(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in range(4):
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        # self.head = nn.Sequential()
        # self.head.fc = nn.Linear(768, 1000)
        # self.neck = nn.AdaptiveAvgPool2d(1)
        self.fpn = FPN_x16(in_channels=[384, 768], out_channel=256)
        self.globalclassifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),  # 1x1 卷积降维
            nn.AdaptiveAvgPool2d((6, 6)),  # 全局池化到1×1
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 2)
        )
        # self.local_global_classifier = LocalGlobalLungClassifier(fpn_channel=256, num_classes=5)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in range(4):
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        return tuple(outs)
    def forward(self, x, rois=None):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[-1]
        local_feat = self.fpn(x_fea)  # {"P1": ..., "P4": ...}
        # 3. 局部六肺区分支
        _clsout = self.globalclassifier(local_feat)  # [B,6,4]
        # return _clsout, x_diff_fea
        return _clsout



    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_fpn_fine_x16_global, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2


class PixelUnshuffle(nn.Module):
    r"""Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
    in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`, where r is a downscale factor.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \times \text{downscale\_factor}^2

    .. math::
        H_{out} = H_{in} \div \text{downscale\_factor}

    .. math::
        W_{out} = W_{in} \div \text{downscale\_factor}

    Examples::

        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = torch.randn(1, 1, 12, 12)
        >>> output = pixel_unshuffle(input)
        >>> print(output.size())
        torch.Size([1, 9, 4, 4])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)
class ConvNeXt_our_roi(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.neck_subfea = nn.Sequential(nn.Conv2d(
                self.channels[-2],
                self.channels[-1],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[-1]),nn.AdaptiveAvgPool2d(1))
        self.local_head = nn.Linear(768, 2)
        self._freeze_stages()


    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
            if i == 2:
                fea_stage2=x

        return tuple(outs), fea_stage2
    def forward(self, x, normalized_rois=None):
        # normalized_rois：[num_rois, 4]（x1, y1, x2, y2），坐标值归一化到[0,1]
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea, fea_stage2 = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        # ROI Align处理
        if normalized_rois is not None:
            # rois形状：[num_rois, 5]（batch_idx, x1, y1, x2, y2）
            # 坐标需要归一化到特征图尺寸
            B, C, H, W = fea_stage2.shape

            # ROI Align参数设置
            pooled_features = roi_align(
                input=fea_stage2,
                boxes=normalized_rois,
                output_size=7,  # 输出特征图尺寸
                spatial_scale=1.0,  # 注意：这里需要设置为1.0因为坐标已经转换到特征图空间
                sampling_ratio=2
            )  # [num_rois, C, 7, 7]

            # 局部分类
            # aa = self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1)

            local_logits = self.local_head(self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1))
            return _clsout, x_diff_fea, local_logits, pooled_features

        # return _clsout
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_roi, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_roi_5classes(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.neck_subfea = nn.Sequential(nn.Conv2d(
                self.channels[-2],
                self.channels[-1],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[-1]),nn.AdaptiveAvgPool2d(1))
        self.local_head = nn.Linear(768, 5)
        self._freeze_stages()


    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
            if i == 2:
                fea_stage2=x

        return tuple(outs), fea_stage2
    def forward(self, x, normalized_rois=None):
        # normalized_rois：[num_rois, 4]（x1, y1, x2, y2），坐标值归一化到[0,1]
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea, fea_stage2 = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        # ROI Align处理
        if normalized_rois is not None:
            # rois形状：[num_rois, 5]（batch_idx, x1, y1, x2, y2）
            # 坐标需要归一化到特征图尺寸
            B, C, H, W = fea_stage2.shape

            # ROI Align参数设置
            pooled_features = roi_align(
                input=fea_stage2,
                boxes=normalized_rois,
                output_size=7,  # 输出特征图尺寸
                spatial_scale=1.0/16,   # 关键参数：将原始坐标映射到特征图，通常为1/下采样步长。
                sampling_ratio=2
            )  # [num_rois, C, 7, 7]

            # 局部分类
            # aa = self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1)

            local_logits = self.local_head(self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1))
            return _clsout, x_diff_fea, local_logits, pooled_features

        # return _clsout
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_roi_5classes, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_roi_5classes_cross(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.neck_subfea = nn.Sequential(nn.Conv2d(
                self.channels[-2],
                self.channels[-1],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[-1]),nn.AdaptiveAvgPool2d(1))
        self.local_head = nn.Linear(768, 5)
        self._freeze_stages()


    def forward_fea(self, x, normalized_rois=None):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
            if i == 2:
                fea_stage2=x
        return tuple(outs), fea_stage2
    def forward(self, x, normalized_rois=None):
        x_fea, fea_stage2 = self.forward_fea(x, normalized_rois)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        # ROI Align处理
        if normalized_rois is not None:
            # ROI Align参数设置
            pooled_features = roi_align(
                input=fea_stage2,
                boxes=normalized_rois,
                output_size=7,  # 输出特征图尺寸
                spatial_scale=1.0/16,   # 关键参数：将原始坐标映射到特征图，通常为1/下采样步长。
                sampling_ratio=2
            )  # [num_rois, C, 7, 7]
            local_logits = self.local_head(self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1))
            return _clsout, x_diff_fea, local_logits, pooled_features

        # return _clsout
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_roi_5classes_cross, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_roi_5classes2(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=[0,1,2,3],
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.global_downsample_layers = ModuleList()
        self.local_downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.global_downsample_layers.append(stem)
        self.local_downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.global_stages = nn.ModuleList()
        self.local_stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.global_downsample_layers.append(downsample_layer)
                self.local_downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.global_stages.append(stage)
            self.local_stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'global_norm{i}', norm_layer)
                self.add_module(f'local_norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.neck_subfea = nn.Sequential(nn.Conv2d(
                self.channels[-2],
                self.channels[-1],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[-1]),nn.AdaptiveAvgPool2d(1))
        self.local_head = nn.Linear(768, 5)
        self._freeze_stages()


    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
            if i == 2:
                fea_stage2=x

        return tuple(outs), fea_stage2

    # normalized_rois：[num_rois, 4]（x1, y1, x2, y2），坐标值归一化到[0,1]
    # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
    # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
    # x_fea=self.fea_diff_forward(x,x_lr)
    def forward(self, x, normalized_rois=None):
        x_fea, fea_stage2 = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        # ROI Align处理
        if normalized_rois is not None:
            # rois形状：[num_rois, 5]（batch_idx, x1, y1, x2, y2）
            # ROI Align参数设置
            pooled_features = roi_align(
                input=fea_stage2,
                boxes=normalized_rois,
                output_size=7,  # 输出特征图尺寸
                spatial_scale=1.0/16,   # 关键参数：将原始坐标映射到特征图，通常为1/下采样步长。
                sampling_ratio=2
            )  # [num_rois, C, 7, 7]

            # 局部分类
            # aa = self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1)

            local_logits = self.local_head(self.neck_subfea(pooled_features).flatten(start_dim=1, end_dim=-1))
            return _clsout, x_diff_fea, local_logits, pooled_features

        # return _clsout
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_roi_5classes_cross, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_gin(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)
        self.gin=GINGroupConv()

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, inference=False):
        if not inference:
            x=self.gin(x)
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout
        # return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            ginlayer=self.gin
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), ginlayer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_gin, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

'''https://github.com/caotongabc/SwinT-SRNet/blob/master/SwinT/HF%2BSwinT/Train-Tiny.py 
SwinT-SRNet: Swin transformer with imagesuper-resolution reconstruction network for pollen images classification
 '''

class ConvNeXt_our_hf(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        # return _clsout, x_diff_fea
        return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_hiera(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_fc = nn.Linear(770, 5)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        _detailed_fea=torch.cat((_clsout,_fea), dim=-1)
        _detailed_clsout = self.head.detailed_fc(_detailed_fea)
        # _clsout_softmax=torch.nn.functional.softmax(_clsout, dim=-1)
        # _detailed_clsout[:,:3]*=_clsout_softmax[:,0].unsqueeze(-1)
        # _detailed_clsout[:, 3:] *= _clsout_softmax[:, 1].unsqueeze(-1)
        # _detailed_clsout=_clsout_softmax*_detailed_clsout
        return _clsout, _detailed_clsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_hiera, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_2_5cls(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        # self.avg_pool_hf = nn.AvgPool2d(8)
        # self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        # self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        # self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_5_fc = nn.Linear(768, 5)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        # x_avg = self.avg_pool_hf(x)
        # x_up = self.upsample_hf(x_avg)
        # x_high_freq = x - x_up
        # x1 = self.conv1_hf(x_high_freq)
        # x3 = self.conv3_hf(x1)
        # x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        # self.x_diff_fea = x_diff_fea
        _fea = x_fea[1]
        # _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        _5clsout = self.head.detailed_5_fc(_fea)
        return _clsout, _5clsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_2_5cls, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_hiera_tree(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        # self.avg_pool_hf = nn.AvgPool2d(8)
        # self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        # self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        # self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_sick_fc = nn.Linear(768, 3)
        self.head.detailed_health_fc = nn.Linear(768, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # outs.append(norm_layer(x))
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    # def forward(self, x_global, x_6locals):
    #     x_fea = self.forward_fea(x_global)
    #     x_diff_fea=x_fea[0]
    #     _fea = x_fea[1]
    #     _global_clsout = self.head.fc(_fea)
    #
    #     x_6local_fea = self.forward_fea(x_6locals)
    #     x_diff_fea = x_6local_fea[0] ##B,C,H,W
    #     _fea = x_6local_fea[1]##B,C,H,W
    #     _6local_clsout = self.head.fc(_fea)
    #     weights = self.head.fc.weight
    #     fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
    #     _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
    #     _6local_detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
    #     _6local_detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
    #     return _global_clsout, _6local_clsout, _6local_detailed_sick_clsout, _6local_detailed_health_clsout, x_diff_fea
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        _fea = x_fea[-1]
        _clsout = self.head.fc(_fea)
        weights = self.head.fc.weight
        fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
        _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)

        # _fea_sick=_fea* weights[0].unsqueeze(0)
        # _fea_health = _fea * weights[1].unsqueeze(0)

        _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
        _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
        # return _clsout
        return _clsout, _fea
        # return _clsout, _fea, x_diff_fea
        # return _clsout, _detailed_sick_clsout, _detailed_health_clsout, _fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_hiera_tree, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

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
class ConvNeXt_our_pcam_hiera_tree(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 num_classes=[1,1],
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_classes=num_classes
        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        # self.avg_pool_hf = nn.AvgPool2d(8)
        # self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        # self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        # self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        for index, num_class in enumerate(self.num_classes):
            setattr(
                self,
                "fc_" +
                str(index),
                nn.Conv2d(
                    768,
                    num_class,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))
            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()
        self.head.detailed_sick_fc = nn.Linear(768, 3)
        self.head.detailed_health_fc = nn.Linear(768, 2)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.global_pool = PcamPool()
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap))

        return tuple(outs)
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        _fea = x_fea[1]
        logit_sick = self.fc_0(_fea)
        logit_health = self.fc_1(_fea)
        _clsout = torch.cat((logit_sick, logit_health), dim=1).squeeze(-1).squeeze(-1)

        logit_map_sick = self.fc_0(x_diff_fea)
        prob_map_sick = torch.sigmoid(logit_map_sick)
        weight_map_sick = prob_map_sick / prob_map_sick.sum(dim=2, keepdim=True) \
            .sum(dim=3, keepdim=True)
        feat_sick = x_diff_fea * weight_map_sick
        logit_map_health = self.fc_1(x_diff_fea)
        prob_map_health = torch.sigmoid(logit_map_health)
        weight_map_health = prob_map_health / prob_map_health.sum(dim=2, keepdim=True) \
            .sum(dim=3, keepdim=True)
        feat_health = x_diff_fea * weight_map_health
        _detailed_sick_clsout = self.head.detailed_sick_fc(feat_sick.sum(dim=-1).sum(dim=-1))
        _detailed_health_clsout = self.head.detailed_health_fc(feat_health.sum(dim=-1).sum(dim=-1))
        return _clsout, _detailed_sick_clsout, _detailed_health_clsout, weight_map_sick, _fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_pcam_hiera_tree, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_gl_hiera_tree(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        # self.avg_pool_hf = nn.AvgPool2d(8)
        # self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        # self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        # self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.localfc = nn.Linear(768, 2)
        self.head.detailed_sick_fc = nn.Linear(768, 3)
        self.head.detailed_health_fc = nn.Linear(768, 2)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))

        return tuple(outs)
    # def forward(self, x_global, x_6locals):
    #     x_fea = self.forward_fea(x_global)
    #     x_diff_fea=x_fea[0]
    #     _fea = x_fea[1]
    #     _global_clsout = self.head.fc(_fea)
    #
    #     x_6local_fea = self.forward_fea(x_6locals)
    #     x_diff_fea = x_6local_fea[0] ##B,C,H,W
    #     _fea = x_6local_fea[1]##B,C,H,W
    #     _6local_clsout = self.head.fc(_fea)
    #     weights = self.head.fc.weight
    #     fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
    #     _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
    #     _6local_detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
    #     _6local_detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
    #     return _global_clsout, _6local_clsout, _6local_detailed_sick_clsout, _6local_detailed_health_clsout, x_diff_fea
    def forward(self, x, local_input=False):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        _fea = x_fea[1]
        if local_input==False:
            _clsout = self.head.fc(_fea)
            return _clsout, x_diff_fea
        else:
            _clsout = self.head.localfc(_fea)
            weights = self.head.localfc.weight
            # fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
            _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
            _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
            _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
            return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea

        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_gl_hiera_tree, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_hiera_tree_pixelshuffle(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        # self.nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         self.channels[0],
        #         kernel_size=stem_patch_size,
        #         stride=stem_patch_size),
        #     build_norm_layer(norm_cfg, self.channels[0]),
        # )
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1 * 32 ** 2, kernel_size=1,
                      stride=1,
                      groups=1, bias=True),
            # nn.ReLU(),
            nn.PixelShuffle(32),
            nn.Sigmoid()
        )

        ### Bilinear Attention Pooling, https://github.com/XiaoLing12138/Adaptive-Dual-Axis-Style-based-Recalibration-Network

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_sick_fc = nn.Linear(768, 3)
        self.head.detailed_health_fc = nn.Linear(768, 2)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def bap_forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix
    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, return_attn=False):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        attn=self.final_up(x_diff_fea)
        # _fea = self.bap_forward(x_diff_fea, attn)
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        weights = self.head.fc.weight
        fea_sick = x_diff_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fea_health = x_diff_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
        _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
        _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
        _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
        # return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea
        # if return_attn:
        #     return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea, attn
        # else:
        #     return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea
        return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_hiera_tree_pixelshuffle, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_hiera_tree_pixelshuffle_causal(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        # self.nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         self.channels[0],
        #         kernel_size=stem_patch_size,
        #         stride=stem_patch_size),
        #     build_norm_layer(norm_cfg, self.channels[0]),
        # )
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1 * 32 ** 2, kernel_size=1,
                      stride=1,
                      groups=1, bias=True),
            # nn.ReLU(),
            nn.PixelShuffle(32),
            nn.Sigmoid()
        )

        ### Bilinear Attention Pooling, https://github.com/XiaoLing12138/Adaptive-Dual-Axis-Style-based-Recalibration-Network

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_sick_fc = nn.Linear(768, 3)
        self.head.detailed_health_fc = nn.Linear(768, 2)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def bap_forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix
    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, return_attn=False):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        attn=self.final_up(x_diff_fea)
        # _fea = self.bap_forward(x_diff_fea, attn)
        scale_factor = x_diff_fea.shape[-1] / attn.shape[-1]
        attn_map = F.interpolate(attn, scale_factor=scale_factor, mode="bicubic")
        enhance_fea=x_diff_fea*attn_map
        if self.training:
            fake_att = torch.zeros_like(attn_map).uniform_(0, 1)
            fake_enhance_fea = x_diff_fea * fake_att
            fake_fea = self.neck(fake_enhance_fea).flatten(start_dim=1, end_dim=-1)
            fake_clsout = self.head.fc(fake_fea)
            weights = self.head.fc.weight
            fake_fea_sick = fake_enhance_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            fake_fea_health = fake_enhance_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            fake_fea_sick = self.neck(fake_fea_sick).flatten(start_dim=1, end_dim=-1)
            fake_fea_health = self.neck(fake_fea_health).flatten(start_dim=1, end_dim=-1)
            fake_detailed_sick_clsout = self.head.detailed_sick_fc(fake_fea_sick)
            fake_detailed_health_clsout = self.head.detailed_health_fc(fake_fea_health)
        _fea = self.neck(enhance_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        weights = self.head.fc.weight
        fea_sick = enhance_fea * weights[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fea_health = enhance_fea * weights[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        _fea_sick = self.neck(fea_sick).flatten(start_dim=1, end_dim=-1)
        _fea_health = self.neck(fea_health).flatten(start_dim=1, end_dim=-1)
        _detailed_sick_clsout = self.head.detailed_sick_fc(_fea_sick)
        _detailed_health_clsout = self.head.detailed_health_fc(_fea_health)
        # if return_attn:
        #     if self.training:
        #         return _clsout,_clsout-fake_clsout, _detailed_sick_clsout, _detailed_sick_clsout-fake_detailed_sick_clsout, _detailed_health_clsout,_detailed_health_clsout-fake_detailed_health_clsout, x_diff_fea, attn
        #     return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea, attn
        # else:
        #     return _clsout, _detailed_sick_clsout, _detailed_health_clsout, x_diff_fea
        return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_hiera_tree_pixelshuffle_causal, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
# class ConvNeXt_our_pixelshuffle(BaseBackbone):
#     arch_settings = {
#         'atto': {
#             'depths': [2, 2, 6, 2],
#             'channels': [40, 80, 160, 320]
#         },
#         'femto': {
#             'depths': [2, 2, 6, 2],
#             'channels': [48, 96, 192, 384]
#         },
#         'pico': {
#             'depths': [2, 2, 6, 2],
#             'channels': [64, 128, 256, 512]
#         },
#         'nano': {
#             'depths': [2, 2, 8, 2],
#             'channels': [80, 160, 320, 640]
#         },
#         'tiny': {
#             'depths': [3, 3, 9, 3],
#             'channels': [96, 192, 384, 768]
#         },
#         'small': {
#             'depths': [3, 3, 27, 3],
#             'channels': [96, 192, 384, 768]
#         },
#         'base': {
#             'depths': [3, 3, 27, 3],
#             'channels': [128, 256, 512, 1024]
#         },
#         'large': {
#             'depths': [3, 3, 27, 3],
#             'channels': [192, 384, 768, 1536]
#         },
#         'xlarge': {
#             'depths': [3, 3, 27, 3],
#             'channels': [256, 512, 1024, 2048]
#         },
#         'huge': {
#             'depths': [3, 3, 27, 3],
#             'channels': [352, 704, 1408, 2816]
#         }
#     }
#
#     def __init__(self,
#                  arch='tiny',
#                  in_channels=3,
#                  stem_patch_size=4,
#                  norm_cfg=dict(type='LN2d', eps=1e-6),
#                  act_cfg=dict(type='GELU'),
#                  linear_pw_conv=True,
#                  use_grn=True,
#                  drop_path_rate=0.2,
#                  layer_scale_init_value=0.,
#                  out_indices=-1,
#                  frozen_stages=0,
#                  gap_before_final_norm=True,
#                  with_cp=False,
#                  init_cfg=[
#                      dict(
#                          type='TruncNormal',
#                          layer=['Conv2d', 'Linear'],
#                          std=.02,
#                          bias=0.),
#                      dict(
#                          type='Constant', layer=['LayerNorm'], val=1.,
#                          bias=0.),
#                  ]):
#         super().__init__(init_cfg=init_cfg)
#
#         if isinstance(arch, str):
#             assert arch in self.arch_settings, \
#                 f'Unavailable arch, please choose from ' \
#                 f'({set(self.arch_settings)}) or pass a dict.'
#             arch = self.arch_settings[arch]
#         elif isinstance(arch, dict):
#             assert 'depths' in arch and 'channels' in arch, \
#                 f'The arch dict must have "depths" and "channels", ' \
#                 f'but got {list(arch.keys())}.'
#
#         self.depths = arch['depths']
#         self.channels = arch['channels']
#         assert (isinstance(self.depths, Sequence)
#                 and isinstance(self.channels, Sequence)
#                 and len(self.depths) == len(self.channels)), \
#             f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
#             'should be both sequence with the same length.'
#
#         self.num_stages = len(self.depths)
#
#         if isinstance(out_indices, int):
#             out_indices = [out_indices]
#         assert isinstance(out_indices, Sequence), \
#             f'"out_indices" must by a sequence or int, ' \
#             f'get {type(out_indices)} instead.'
#         for i, index in enumerate(out_indices):
#             if index < 0:
#                 out_indices[i] = 4 + index
#                 assert out_indices[i] >= 0, f'Invalid out_indices {index}'
#         self.out_indices = out_indices
#
#         self.frozen_stages = frozen_stages
#         self.gap_before_final_norm = gap_before_final_norm
#         self.avg_pool_hf = nn.AvgPool2d(8)
#         self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
#         self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
#         self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
#         # stochastic depth decay rule
#         dpr = [
#             x.item()
#             for x in torch.linspace(0, drop_path_rate, sum(self.depths))
#         ]
#         block_idx = 0
#
#         # 4 downsample layers between stages, including the stem layer.
#         self.downsample_layers = ModuleList()
#         stem = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 self.channels[0],
#                 kernel_size=stem_patch_size,
#                 stride=stem_patch_size),
#             build_norm_layer(norm_cfg, self.channels[0]),
#         )
#         self.downsample_layers.append(stem)
#
#         # 4 feature resolution stages, each consisting of multiple residual
#         # blocks
#         self.stages = nn.ModuleList()
#
#         for i in range(self.num_stages):
#             depth = self.depths[i]
#             channels = self.channels[i]
#
#             if i >= 1:
#                 downsample_layer = nn.Sequential(
#                     build_norm_layer(norm_cfg, self.channels[i - 1]),
#                     nn.Conv2d(
#                         self.channels[i - 1],
#                         channels,
#                         kernel_size=2,
#                         stride=2),
#                 )
#                 self.downsample_layers.append(downsample_layer)
#
#             stage = Sequential(*[
#                 ConvNeXtBlock(
#                     in_channels=channels,
#                     drop_path_rate=dpr[block_idx + j],
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     linear_pw_conv=linear_pw_conv,
#                     layer_scale_init_value=layer_scale_init_value,
#                     use_grn=use_grn,
#                     with_cp=with_cp) for j in range(depth)
#             ])
#             block_idx += depth
#
#             self.stages.append(stage)
#
#             if i in self.out_indices:
#                 norm_layer = build_norm_layer(norm_cfg, channels)
#                 self.add_module(f'norm{i}', norm_layer)
#
#         # self.nn.Sequential(
#         #     nn.Conv2d(
#         #         in_channels,
#         #         self.channels[0],
#         #         kernel_size=stem_patch_size,
#         #         stride=stem_patch_size),
#         #     build_norm_layer(norm_cfg, self.channels[0]),
#         # )
#         self.final_up = nn.Sequential(
#             nn.Conv2d(in_channels=768, out_channels=1 * 32 ** 2, kernel_size=1,
#                       stride=1,
#                       groups=1, bias=True),
#             nn.ReLU(),
#             nn.PixelShuffle(32),
#         )
#
#         ### Bilinear Attention Pooling, https://github.com/XiaoLing12138/Adaptive-Dual-Axis-Style-based-Recalibration-Network
#
#         self.head = nn.Sequential()
#         self.head.fc = nn.Linear(768, 1000)
#         self.neck = nn.AdaptiveAvgPool2d(1)
#         self._freeze_stages()
#
#     def bap_forward(self, features, attentions):
#         B, C, H, W = features.size()
#         _, M, AH, AW = attentions.size()
#
#         # match size
#         if AH != H or AW != W:
#             attentions = F.upsample_bilinear(attentions, size=(H, W))
#
#         feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
#
#         # sign-sqrt
#         feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)
#
#         # l2 normalization along dimension M and C
#         feature_matrix = F.normalize(feature_matrix, dim=-1)
#         return feature_matrix
#     def forward_fea(self, x):
#         x_avg = self.avg_pool_hf(x)
#         x_up = self.upsample_hf(x_avg)
#         x_high_freq = x - x_up
#         x1 = self.conv1_hf(x_high_freq)
#         x3 = self.conv3_hf(x1)
#         x = x3 + x
#         outs = []
#         for i, stage in enumerate(self.stages):
#             x = self.downsample_layers[i](x)
#             x = stage(x)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'norm{i}')
#                 outs.append(norm_layer(x))
#
#         return tuple(outs)
#     def forward(self, x, return_attn=False):
#         # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
#         # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
#         # x_fea=self.fea_diff_forward(x,x_lr)
#         x_fea = self.forward_fea(x)
#         # x_lr_fea = self.fea_diff_forward(x_lr)
#         x_diff_fea=x_fea[0]
#         # 捕获 x_diff_fea 激活
#         self.x_diff_fea = x_diff_fea
#         attn=self.final_up(x_diff_fea)
#         # _fea = self.bap_forward(x_diff_fea, attn)
#         _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
#         _clsout = self.head.fc(_fea)
#         if return_attn:
#             return _clsout, x_diff_fea, attn
#         else:
#             return _clsout, x_diff_fea
#         # return _clsout
#     def _freeze_stages(self):
#         for i in range(self.frozen_stages):
#             downsample_layer = self.downsample_layers[i]
#             stage = self.stages[i]
#             downsample_layer.eval()
#             stage.eval()
#             for param in chain(downsample_layer.parameters(),
#                                stage.parameters()):
#                 param.requires_grad = False
#
#     def train(self, mode=True):
#         super(ConvNeXt_our_pixelshuffle, self).train(mode)
#         self._freeze_stages()
#
#     def get_layer_depth(self, param_name: str, prefix: str = ''):
#         """Get the layer-wise depth of a parameter.
#
#         Args:
#             param_name (str): The name of the parameter.
#             prefix (str): The prefix for the parameter.
#                 Defaults to an empty string.
#
#         Returns:
#             Tuple[int, int]: The layer-wise depth and the num of layers.
#         """
#
#         max_layer_id = 12 if self.depths[-2] > 9 else 6
#
#         if not param_name.startswith(prefix):
#             # For subsequent module like head
#             return max_layer_id + 1, max_layer_id + 2
#
#         param_name = param_name[len(prefix):]
#         if param_name.startswith('downsample_layers'):
#             stage_id = int(param_name.split('.')[1])
#             if stage_id == 0:
#                 layer_id = 0
#             elif stage_id == 1 or stage_id == 2:
#                 layer_id = stage_id + 1
#             else:  # stage_id == 3:
#                 layer_id = max_layer_id
#
#         elif param_name.startswith('stages'):
#             stage_id = int(param_name.split('.')[1])
#             block_id = int(param_name.split('.')[2])
#             if stage_id == 0 or stage_id == 1:
#                 layer_id = stage_id + 1
#             elif stage_id == 2:
#                 layer_id = 3 + block_id // 3
#             else:  # stage_id == 3:
#                 layer_id = max_layer_id
#
#         # final norm layer
#         else:
#             layer_id = max_layer_id + 1
#
#         return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_2classifier(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.sub_cls = nn.Linear(768, 2)
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, sub_input=False):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        if sub_input:
            _clsout = self.sub_cls(_fea)
        else:
            _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_2classifier, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_our_hf_subfeacls(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)

        self.cls_subfea = nn.Linear(768, 2)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.subneck = nn.AdaptiveAvgPool2d((3,2))
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        _subfea=self.subneck(x_diff_fea)
        b,c,h,w=_subfea.shape
        _subfea=_subfea.view(_subfea.shape[0], _subfea.shape[1], -1)
        # 步骤 2: 转置为 [B, H*W, C]
        transposed_tensor = _subfea.transpose(1, 2)
        # 步骤 3: 最终变形为 [B*H*W, C]
        final_reshaped_tensor = transposed_tensor.contiguous().view(-1, c)
        _subclsout = self.cls_subfea(final_reshaped_tensor)
        # restored_tensor = _subclsout.view(b, h * w, -1)
        # 步骤 2: 转置回 [B, C, H*W]
        # transposed_restored_tensor = restored_tensor.transpose(1, 2)
        return _clsout, _subclsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_subfeacls, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
'''style transfer, 用于低层特征风格一致性损失计算'''
class ConvNeXt_our_hf_style(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i ==0:
                x0=x.clone()
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs), x0
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea, x0 = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea, x0
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_style, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
'''变异系数，标准差与均值的比值'''
class ConvNeXt_our_hf_cv(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        B, C, H, W = x.shape
        # 使用展开操作 (unfold) 提取滑动窗口
        window_size=3
        # 使用展开操作 (unfold) 提取滑动窗口
        unfolded = x.unfold(2, window_size, 1).unfold(3, window_size, 1)
        # unfolded 形状: (B, C, H_out, W_out, window_size, window_size)
        # 其中 H_out = H - window_size + 1, W_out = W - window_size + 1

        # 调整形状以便计算
        unfolded = unfolded.contiguous().view(B, C, H - window_size + 1, W - window_size + 1, -1)
        # unfolded 形状: (B, C, H_out, W_out, window_size * window_size)

        # 计算均值和标准差
        mean = torch.mean(unfolded, dim=-1)  # 均值，形状: (B, C, H_out, W_out)
        std = torch.std(unfolded, dim=-1, unbiased=False)  # 标准差，形状: (B, C, H_out, W_out)

        # 计算变异系数
        cv = std / (mean + 1e-6)

        # 边界填充，使输出特征图大小与输入一致
        pad_h = window_size // 2
        pad_w = window_size // 2
        cv_padded = torch.nn.functional.pad(cv, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x + cv_padded
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_cv, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
'''迭代增强特征'''
class ConvNeXt_our_hf_curenhance(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, xorg):
        h=xorg.shape[-1]
        x_avg = self.avg_pool_hf(xorg)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = xorg - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + xorg
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        weights = self.head.fc.weight
        selected_weights_sick = weights[0]  # 形状：(batch_size, 2048)
        # 计算 CAM
        cam_sick = torch.matmul(x.permute(0, 2, 3, 1), selected_weights_sick.unsqueeze(-1))
        cam_sick=1-torch.sigmoid(cam_sick).permute(0, 3, 1, 2)
        cammap = F.interpolate(cam_sick, scale_factor=h/cam_sick.shape[2])

        xx=xorg+xorg*cammap
        x_avg = self.avg_pool_hf(xx)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = xx - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + xx
        outs2 = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs2.append(norm_layer(x))
        outs[-1]=outs[-1]+outs2[-1]
        return tuple(outs)
    def forward(self, x):
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_our_hf_curenhance, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
class ConvNeXt_gin_hf(BaseBackbone):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.2,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)
        self.gin=GINGroupConv()

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages()

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))

        return tuple(outs)
    def forward(self, x, inference=False):
        if not inference:
            x=self.gin(x)
        # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
        # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
        # x_fea=self.fea_diff_forward(x,x_lr)
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            ginlayer=self.gin
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), ginlayer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_gin_hf, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
def convnexttiny(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_org(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_org_pooling(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our_pooling(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_org_pixelshuffle(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our_pixelshuffle(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_fpn_local_global(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_fpn_local_global(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_fpn_fine_local_global(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_fpn_fine_local_global(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_fpn_x8_fine_global(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_fpn_fine_global(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_fpn_x16_fine_global(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_fpn_fine_x16_global(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_hiera(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf_hiera(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_hiera_tree(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf_hiera_tree(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_pcam_hiera_tree(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_pcam_hiera_tree(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_gl_hiera_tree(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_gl_hiera_tree(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_2_5cls(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_2_5cls(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_hiera_tree_pixelshuffleattn(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model =ConvNeXt_our_hf_hiera_tree_pixelshuffle(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_hiera_tree_pixelshuffleattn_causal(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model =ConvNeXt_our_hf_hiera_tree_pixelshuffle_causal(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_attn(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_pixelshuffle(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_2classifiers(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf_2classifier(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_nohf(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_roi(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our_roi(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def convnexttiny_roi_5classes(pretrained=False, drop_path_rate=0.2):
    model = ConvNeXt_our_roi_5classes(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_style(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf_style(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
def convnexttiny_subfeacls(pretrained=False, drop_path_rate=0.2):
    # model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    model = ConvNeXt_our_hf_subfeacls(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf_curenhance(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
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

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
if __name__ == '__main__':
    import time
    from mmpretrain import get_model
    device = torch.device('cuda:0')
    pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth'
    model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
    model.head.fc = nn.Linear(768, 2)
    # model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
    # model.head.fc = nn.Linear(768, 2)
    model.to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.5fM" % (total / 1e6))
    x = torch.rand(1, 3, 224, 224).to(device)
    model.eval()
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        _ = model(x)
        timer.toc()
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))
    b=1