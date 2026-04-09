from mmpretrain.models.backbones.convnext import *
from model_cls.image_filter import *
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
                 up_scale=4,
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
        self.up_scale=up_scale
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
        self.upsample_layers = ModuleList()
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


        self.dec_stages = nn.ModuleList()
        dec_depths=self.depths[::-1]
        dec_channels = self.channels[::-1]
        self.skip_conn_concat = nn.ModuleList()
        # block_idx = 0
        for i in range(self.num_stages):
            depth = dec_depths[i]
            channels = dec_channels[i]
            if i<self.num_stages-1:
                upsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, channels),
                    nn.ConvTranspose2d(in_channels=channels, out_channels=dec_channels[i+1], kernel_size=2,
                                       stride=2,
                                       padding=0),
                )
                self.upsample_layers.append(upsample_layer)
                self.skip_conn_concat.append(nn.Conv2d(
                    dec_channels[i + 1]*2,
                    dec_channels[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1),
                )

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx - j-1],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx -= depth

            self.dec_stages.append(stage)
        self.upsample_layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=dec_channels[-1], out_channels=dec_channels[-1], kernel_size=2, stride=2,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dec_channels[-1], out_channels=dec_channels[-1], kernel_size=3, stride=1,
                      padding=1),  # 392*392*128->390*390*64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=dec_channels[-1], out_channels=dec_channels[-1] // 2, kernel_size=2, stride=2,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dec_channels[-1] // 2, out_channels=dec_channels[-1] // 2, kernel_size=3, stride=1,
                      padding=1),  # 392*392*128->390*390*64
            nn.ReLU(inplace=True),
        ))
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels=dec_channels[-1] // 2, out_channels=3 * self.up_scale ** 2, kernel_size=3, padding=1,
                      stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(self.up_scale)
        )
        self.head = nn.Sequential()
        self.head.fc = nn.Linear(768, 1000)
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_hf = nn.AvgPool2d(8)
        self.upsample_hf = nn.Upsample(scale_factor=8, mode='bicubic')
        self.conv1_hf = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv3_hf = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self._freeze_stages()

    def forward_srfea(self, x):
        outs = []
        encs=[]
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            encs.append(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x))
        decs=[]
        for i, stage in enumerate(self.dec_stages):
            x = stage(x)
            x = self.upsample_layers[i](x)
            if i< len(self.dec_stages)-1:
                x = torch.cat([x, encs[-(i + 2)]], dim=1)
                x = self.skip_conn_concat[i](x)
            decs.append(x)
        sr = self.final_up(x)
        return sr
    def forward_sr(self, inp):
        x = F.interpolate(inp, scale_factor=1 / self.up_scale, mode='bilinear')
        inp_hr = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear')
        x_sr = self.forward_srfea(x)
        x_sr = x_sr + inp_hr
        return x_sr, x
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
        x_fea = self.forward_fea(x)
        # x_lr_fea = self.fea_diff_forward(x_lr)
        x_diff_fea = x_fea[0]
        # 捕获 x_diff_fea 激活
        self.x_diff_fea = x_diff_fea
        _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
        _clsout = self.head.fc(_fea)
        return _clsout, x_diff_fea
    # def forward(self, x):
    #     # lr = F.interpolate(x, scale_factor=0.25, mode="nearest")
    #     # x_lr = F.interpolate(lr, scale_factor=4, mode="nearest")
    #     # x_fea=self.fea_diff_forward(x,x_lr)
    #     x_fea = self.forward_fea(x)
    #     # x_lr_fea = self.fea_diff_forward(x_lr)
    #     x_diff_fea=x_fea[0]
    #     # 捕获 x_diff_fea 激活
    #     self.x_diff_fea = x_diff_fea
    #     _fea = self.neck(x_diff_fea).flatten(start_dim=1, end_dim=-1)
    #     _clsout = self.head.fc(_fea)
    #     return _clsout
    #     # return _clsout, x_diff_fea
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
    model = ConvNeXt_our(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_our_hf(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin(arch='tiny', drop_path_rate=drop_path_rate)
    # model = ConvNeXt_gin_hf(arch='tiny', drop_path_rate=drop_path_rate)
    if pretrained:
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        state_dict = torch.load(pretrained_cfg, map_location='cpu')['state_dict']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

if __name__ == '__main__':
    model=convnexttiny(pretrained=True, drop_path_rate=0.0)
    aa=torch.rand(1,3,512,512)
    y=model.forward_sr(aa)
    print(y.shape)
    b=1