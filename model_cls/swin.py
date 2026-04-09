from mmpretrain.models.backbones.swin_transformer import *
class SwinTransformer_our(BaseBackbone):
    """Swin Transformer.

    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>`_

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

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
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
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SwinTransformer
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'expansion_ratio': 3}))
        >>> self = SwinTransformer(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

    _version = 3
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 out_indices=(3, ),
                 out_after_downsample=False,
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=True,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer_our, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

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

        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            # self.add_module(f'norm{i}', norm_layer)
            self.add_module(f'norm', norm_layer)

    def init_weights(self):
        super(SwinTransformer_our, self).init_weights()

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
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
        # return _clsout

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
                or version < 2) and self.__class__ is SwinTransformer:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None
                or version < 3) and self.__class__ is SwinTransformer:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)

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
        super(SwinTransformer_our, self).train(mode)
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

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_bias_table_pretrained = state_dict[ckpt_key]
                relative_position_bias_table_current = state_dict_model[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if L1 != L2:
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)
                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size,
                        relative_position_bias_table_pretrained, nH1)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    del state_dict[index_buffer]

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = sum(self.depths) + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = param_name.split('.')[3]
            if block_id in ('reduction', 'norm'):
                layer_depth = sum(self.depths[:stage_id + 1])
            else:
                layer_depth = sum(self.depths[:stage_id]) + int(block_id) + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers

class SwinTransformer_our_2_5cls(BaseBackbone):
    """Swin Transformer.

    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>`_

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

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
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
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SwinTransformer
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'expansion_ratio': 3}))
        >>> self = SwinTransformer(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

    _version = 3
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 out_indices=(3, ),
                 out_after_downsample=False,
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=True,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer_our_2_5cls, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

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

        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.head.detailed_5_fc = nn.Linear(768, 5)
        self.neck = nn.AdaptiveAvgPool2d(1)

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            # self.add_module(f'norm{i}', norm_layer)
            self.add_module(f'norm', norm_layer)

    def init_weights(self):
        super(SwinTransformer_our_2_5cls, self).init_weights()

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
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        _5clsout = self.head.detailed_5_fc(_fea)
        return _clsout, _5clsout, x_diff_fea
        # return _clsout

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
                or version < 2) and self.__class__ is SwinTransformer:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None
                or version < 3) and self.__class__ is SwinTransformer:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)

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
        super(SwinTransformer_our_2_5cls, self).train(mode)
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

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_bias_table_pretrained = state_dict[ckpt_key]
                relative_position_bias_table_current = state_dict_model[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if L1 != L2:
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)
                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size,
                        relative_position_bias_table_pretrained, nH1)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    del state_dict[index_buffer]

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = sum(self.depths) + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = param_name.split('.')[3]
            if block_id in ('reduction', 'norm'):
                layer_depth = sum(self.depths[:stage_id + 1])
            else:
                layer_depth = sum(self.depths[:stage_id]) + int(block_id) + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
class SwinTransformer_our_hf(BaseBackbone):
    """Swin Transformer.

    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>`_

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

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
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
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SwinTransformer
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'expansion_ratio': 3}))
        >>> self = SwinTransformer(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

    _version = 3
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 out_indices=(3, ),
                 out_after_downsample=False,
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=True,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer_our_hf, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

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

        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            # self.add_module(f'norm{i}', norm_layer)
            self.add_module(f'norm', norm_layer)

    def init_weights(self):
        super(SwinTransformer_our_hf, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward_fea(self, x):
        x_avg = self.avg_pool_hf(x)
        x_up = self.upsample_hf(x_avg)
        x_high_freq = x - x_up
        x1 = self.conv1_hf(x_high_freq)
        x3 = self.conv3_hf(x1)
        x = x3 + x
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)
    def forward(self, x):
        x_fea = self.forward_fea(x)
        x_diff_fea=x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
                or version < 2) and self.__class__ is SwinTransformer:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None
                or version < 3) and self.__class__ is SwinTransformer:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)

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
        super(SwinTransformer_our_hf, self).train(mode)
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

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_bias_table_pretrained = state_dict[ckpt_key]
                relative_position_bias_table_current = state_dict_model[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if L1 != L2:
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)
                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size,
                        relative_position_bias_table_pretrained, nH1)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    del state_dict[index_buffer]

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = sum(self.depths) + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = param_name.split('.')[3]
            if block_id in ('reduction', 'norm'):
                layer_depth = sum(self.depths[:stage_id + 1])
            else:
                layer_depth = sum(self.depths[:stage_id]) + int(block_id) + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
def swintiny(pretrained=False, drop_path_rate=0.2):
    model = SwinTransformer_our(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace("norm.", 'norm3.'): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def swintiny_2_5cls(pretrained=False, drop_path_rate=0.2):
    model = SwinTransformer_our_2_5cls(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace("norm.", 'norm3.'): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def swintiny_hf(pretrained=False, drop_path_rate=0.2):
    model = SwinTransformer_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace("norm.", 'norm3.'): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
if __name__ == '__main__':
    model=swintiny(pretrained=True)
    aa=model.eval()
    aa=torch.rand(1,3,224,224)
    y=model(aa)
    b=1