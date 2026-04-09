 

from .mambaout import MambaOut
# from .build import build_model as build_model_swin
import torch
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE


    if model_type in ["Mambaout"]:
        model = MambaOut(num_classes=2,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 576])
        # state_dict = torch.hub.load_state_dict_from_url(
        #     url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        # model.load_state_dict(state_dict)
        return model
    return None
    # return build_model_swin(config, is_pretrain)


