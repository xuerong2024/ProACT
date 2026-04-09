import torch
import torch
from safetensors import safe_open
from torch import Tensor, nn
def safetensors_to_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt") as ckpt_file:
        for key in ckpt_file.keys():
            state_dict[key] = ckpt_file.get_tensor(key)
    return state_dict


def rad_dino_gh():
    rad_dino_gh = torch.hub.load(
        '/disk1/wjr/.cache/torch/hub/facebookresearch_dinov2_main',
        'dinov2_vitb14',
        source='local'
    )
    backbone_state_dict = safetensors_to_state_dict("/disk3/wjr/workspace/sec_proj4/rad_dinopt/backbone_compatible.safetensors")
    rad_dino_gh.load_state_dict(backbone_state_dict, strict=True)
    return rad_dino_gh

if __name__ == '__main__':
    rad_dino_gh = rad_dino_gh()
    rad_dino_gh.head=nn.Linear(768, 2)
    x=torch.rand(2,3,224,224).cuda()
    rad_dino_gh.cuda()
    y=rad_dino_gh(x)
    print(y.shape)
