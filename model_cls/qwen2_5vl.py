from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj
class qwen2_5vision(nn.Module):
    # def __init__(self, pth_path='/disk3/wjr/workspace/sec_proj4/qwen2_5VL_3B_visual.pth', n_class=2, patch_size=14, temporal_patch_size=2, merge_size=2):
    def __init__(self, pth_path='/disk3/wjr/workspace/sec_proj4/qwen2_5VL_7B_visual.pth', n_class=2, patch_size=14,
                     temporal_patch_size=2, merge_size=2):

        super().__init__()
        pretrained=torch.load(pth_path, map_location='cpu',weights_only=False)  # 明确允许加载任意对象
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(pretrained['config'])
        self.visual.load_state_dict(pretrained['state_dict'])
        self.temporal_patch_size=temporal_patch_size
        self.merge_size=merge_size
        self.patch_size=patch_size
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.dtype = self.visual.dtype
        # '''3B'''
        # self.clshead=nn.Linear(2048, n_class).type(self.dtype)
        '''7B'''
        self.clshead = nn.Linear(3584, n_class).type(self.dtype)

    def unpack_features(self, output: torch.Tensor, grid_thw: torch.Tensor):
        """
        将 packed 格式的视觉特征还原为按图像分开的 list。

        Args:
            output: (Total_N, D)
            grid_thw: (num_images, 3) -> [T, H, W]

        Returns:
            List[torch.Tensor]，每个元素是 (T*H*W, D)
        """
        features = []
        cumsum = 0
        for t, h, w in grid_thw:
            num_tokens = t * h * w
            img_feat = output[cumsum: cumsum + num_tokens]
            features.append(img_feat)
            cumsum += num_tokens
        return features
    def forward(self, x):
        resized_height, resized_width = x.shape[-2:]
        if x.ndim == 4:
            # add a temporal dimension if we have images
            x = x.unsqueeze(1)
        if x.shape[1] % self.temporal_patch_size != 0:
            repeats = x[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
            x = torch.cat([x, repeats], dim=1)
        batch_size, grid_t, channel = x.shape[:3]
        grid_t = grid_t // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        x = x.view(
            batch_size,
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        x = x.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * batch_size)
        feature=self.visual(x, grid_thw)
        t, h, w = grid_thw[0]  # 所有图像相同，取第一个即可

        output_batch = feature.reshape(batch_size*t, h// self.visual.spatial_merge_size,w // self.visual.spatial_merge_size, feature.shape[-1])
        # feature=self.unpack_features(feature, grid_thw)
        _fea = self.neck(output_batch.permute(0,3,1,2)).flatten(start_dim=1, end_dim=-1)
        _clsout = self.clshead(_fea)
        return _clsout

class qwen2_5vision_lora(nn.Module):
    def __init__(self, pth_path='/disk3/wjr/workspace/sec_proj4/qwen2_5VL_3B_visual.pth', n_class=2, patch_size=14, temporal_patch_size=2, merge_size=2):
        super().__init__()
        self.model=qwen2_5vision(pth_path=pth_path, n_class=n_class, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)
        # pretrained = torch.load(pth_path, map_location='cpu', weights_only=False)  # 明确允许加载任意对象
        # self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(pretrained['config'])
        # self.visual.load_state_dict(pretrained['state_dict'])

        target_modules = self.get_target_modules(self.model.visual)
        lora_kwargs = {
            'r': 8,
            'target_modules': target_modules,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'bias': 'none',
            'modules_to_save': [],
            'use_rslora': False,
            'use_dora': False,
            'lorap_lr_ratio': None,
            'init_lora_weights': True,
        }
        lora_config = LoraConfig(lora_dtype=None, **lora_kwargs)
        self.model.visual=get_peft_model(model=self.model.visual, peft_config=lora_config)
        trainable_params = 0
        all_params=0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if param.dtype == torch.float16:
                    print(
                        f'Converting trainable parameter to fp32: {name} | Shape: {param.shape} | Current dtype: {param.dtype}')
                    param.data = param.data.to(dtype=torch.float32)
                # print(
                #     f'Converting trainable parameter to fp32: {name} | Shape: {param.shape} | Current dtype: {param.dtype}')
                # param.data = param.data.to(dtype=torch.float32)
        print(f"可训练参数量: {trainable_params / 1e6:.4f}M")
        print(f"总参数量: {all_params / 1e6:.4f}M")
        # a=1

    def get_target_modules(self, model):
        """Replace all-linear to actual modules"""
        # model_meta = model.model_meta
        target_modules = []
        # a=find_all_linears(model)
        target_modules += self.find_all_linears(model)
        return target_modules

    def find_all_linears(self, model, model_arch=None, extra_layers=None, sub_module=None):
        lm_head_name = 'lm_head'
        # 'score', 'classifier': classification model
        # 'v_head': reward model
        ignore_layers = [lm_head_name, 'score', 'v_head', 'classifier'] + ['lora_A', 'lora_B', 'base_layer']
        ignore_linear_cls = [
            'glulinear'  # phi4-mm
        ]

        def _cond(name, module):
            module_name = module.__class__.__name__.lower()
            if (extra_layers and isinstance(module, tuple(extra_layers)) or
                ('linear' in module_name and all(linear_cls not in module_name
                                                 for linear_cls in ignore_linear_cls))) and all(layer not in name
                                                                                                for layer in
                                                                                                ignore_layers):
                return True
            return False

        return self.find_layers(model, _cond, sub_module=sub_module)

    def find_layers(self,
                        model,
                        cond,
                        sub_module= None,
                        min_name_len= None,
                        ):
        # The content of target_module_names cannot exist in inner_nodes.
        sub_module_str = sub_module
        if sub_module is None:
            sub_module = model
        else:
            sub_module = deep_getattr(model, sub_module)
        inner_nodes = set()
        for name, module in model.named_modules():
            name = re.sub(r'\d+\.', '{}.', name)
            if not cond(name, module):
                inner_nodes.add(name)
        target_module_names = set()
        for name, module in sub_module.named_modules():
            if sub_module_str:
                name = f'{sub_module_str}.{name}' if name else sub_module_str
            if cond(name, module):
                module_name_list = name.split('.')
                module_name = module_name_list.pop()
                i = 1
                for inner_node in inner_nodes:
                    while module_name_list and inner_node.endswith(re.sub(
                            r'\d+\.', '{}.', module_name)) or min_name_len and i < min_name_len:
                        module_name = f'{module_name_list.pop()}.{module_name}'
                        i += 1
                target_module_names.add(name)
        return list(target_module_names)
    def unpack_features(self, output: torch.Tensor, grid_thw: torch.Tensor):
        """
        将 packed 格式的视觉特征还原为按图像分开的 list。

        Args:
            output: (Total_N, D)
            grid_thw: (num_images, 3) -> [T, H, W]

        Returns:
            List[torch.Tensor]，每个元素是 (T*H*W, D)
        """
        features = []
        cumsum = 0
        for t, h, w in grid_thw:
            num_tokens = t * h * w
            img_feat = output[cumsum: cumsum + num_tokens]
            features.append(img_feat)
            cumsum += num_tokens
        return features
    def forward(self, x):
        _clsout = self.model(x)
        return _clsout
from peft import get_peft_model
from model_cls.qwen2_5.peft import LoraConfig, get_peft_model
if __name__ == "__main__":
    import os
    ckp_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision/best_224_batch16_lr1e-05_testall/model_best_acc_alltest.pth'
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)['model']
    ckp_path2 = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_lora/224_batch16_lr1e-05_testall/model_best_acc_alltest.pth'
    checkpoint2 = torch.load(ckp_path2, map_location="cpu", weights_only=False)['model']
    model = qwen2_5vision_lora(n_class=2).cuda()
    input = torch.ones((3, 3, 504, 504)).cuda()
    y = model(input)
    print(y.shape)

