from mmpretrain.models.backbones.swin_transformer_v2 import *
class SwinTransformerV2(BaseBackbone):
    """Swin Transformer V2.

    A PyTorch implement of : `Swin Transformer V2:
    Scaling Up Capacity and Resolution
    <https://arxiv.org/abs/2111.09883>`_

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of heads in attention
              modules of each stage.
            - **extra_norm_every_n_blocks** (int): Add extra norm at the end
              of main branch every n blocks.

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int | Sequence): The height and width of the window.
            Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        interpolate_mode (str): Select the interpolate mode for absolute
            position embeding vector resize. Defaults to "bicubic".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``
        stage_cfgs (Sequence[dict] | dict): Extra config dict for each
            stage. Defaults to an empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of
            each layer.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SwinTransformerV2
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'padding': 'same'}))
        >>> self = SwinTransformerV2(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48],
                         'extra_norm_every_n_blocks': 0}),
        # head count not certain for huge, and is employed for another
        # parallel study about self-supervised learning.
        **dict.fromkeys(['h', 'huge'],
                        {'embed_dims': 352,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [8, 16, 32, 64],
                         'extra_norm_every_n_blocks': 6}),
        **dict.fromkeys(['g', 'giant'],
                        {'embed_dims': 512,
                         'depths':     [2,  2, 42,  4],
                         'num_heads':  [16, 32, 64, 128],
                         'extra_norm_every_n_blocks': 6}),
    }  # yapf: disable

    _version = 1
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=256,
                 patch_size=4,
                 in_channels=3,
                 window_size=8,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 out_indices=(3, ),
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 pretrained_window_sizes=[0, 0, 0, 0],
                 init_cfg=None):
        super(SwinTransformerV2, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads',
                'extra_norm_every_n_blocks'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.extra_norm_every_n_blocks = self.arch_settings[
            'extra_norm_every_n_blocks']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages
        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)

        if isinstance(window_size, int):
            self.window_sizes = [window_size for _ in range(self.num_layers)]
        elif isinstance(window_size, Sequence):
            assert len(window_size) == self.num_layers, \
                f'Length of window_sizes {len(window_size)} is not equal to '\
                f'length of stages {self.num_layers}.'
            self.window_sizes = window_size
        else:
            raise TypeError('window_size should be a Sequence or int.')

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        if self.use_abs_pos_embed:
            num_patches = self.patch_resolution[0] * self.patch_resolution[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims))
            self._register_load_state_dict_pre_hook(
                self._prepare_abs_pos_embed)

        self._register_load_state_dict_pre_hook(self._delete_reinit_params)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i > 0 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': self.window_sizes[i],
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'extra_norm_every_n_blocks': self.extra_norm_every_n_blocks,
                'pretrained_window_size': pretrained_window_sizes[i],
                'downsample_cfg': dict(use_post_norm=True),
                **stage_cfg
            }

            stage = SwinBlockV2Sequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, embed_dims[i + 1])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

    def init_weights(self):
        super(SwinTransformerV2, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward_fea(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1,
                                                           2).contiguous()
                outs.append(out)

        return tuple(outs)
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SwinTransformerV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'absolute_pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.absolute_pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                'Resize the absolute_pos_embed shape from '
                f'{ckpt_pos_embed_shape} to {self.absolute_pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    def _delete_reinit_params(self, state_dict, prefix, *args, **kwargs):
        # delete relative_position_index since we always re-init it
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        logger.info(
            'Delete `relative_position_index` and `relative_coords_table` '
            'since we always re-init these params according to the '
            '`window_size`, which might cause unwanted but unworried '
            'warnings when loading checkpoint.')
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_coords_table' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]


def swintinyv2(pretrained=False, drop_path_rate=0.2):
    model = SwinTransformerV2(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace("norm.", 'norm3.'): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

if __name__ == '__main__':
    model=swintinyv2(pretrained=True)
    aa=model.eval()
    aa=torch.rand(1,3,512,512)
    y=model(aa)
    b=1