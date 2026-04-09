'''Efficient Test-Time Adaptation of Vision-Language Models, cvpr2023'''
import os
import time
import json
import random
import argparse
import datetime
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 添加自定义路径到全局模块搜索路径中
sys.path.append('/disk3/wjr/workspace/sec_proj4/proj4_feiqu_baseline/')
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from loss.focal_loss import *
from utils.configv2 import get_config
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import *
from mmpretrain import get_model
from utils.datasets_pneum import *
print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)
import utils.misc as misc
from model_cls.pvt import *
# from model_cls.qwen2_5vl import qwen2_5vision
from model_cls.qwen2_5.qwen2_5vl import qwen2_5vision, qwen2_5vision_lora
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:2')
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #将CuDNN的性能优化模式设置为关闭
    cudnn.benchmark = False
    #将CuDNN的确定性模式设置为启用,确保CuDNN在相同的输入下生成相同的输出
    cudnn.deterministic = True
    #CuDNN加速
    cudnn.enabled = True
    # 2. 禁用CUDA的非确定性操作
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)
def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--model_name', type=str,
                        default='resnet50',
                        help='t')
    # easy config modification
    parser.add_argument('--batch-size', type=int, default=2, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--base_lr', type=float, default=1e-5)

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--data_root', type=str, default='/disk3/wjr/dataset/nejm/shanxidataset/', help='The path of dataset')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    args, unparsed = parser.parse_known_args()
    return args
from torchvision import transforms as pth_transforms
from PIL import ImageFilter, ImageOps
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL save.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img1, img2, mask2=None, sub_rois=None):
        do_it = random.random() <= self.prob
        if not do_it:
            return img1, img2, mask2, sub_rois
        return img1.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        ), img2, mask2, sub_rois
from data import transforms
# from torchvision import transforms as pth_transforms

def build_phe_loader(args):
    # 设置随机数种子
    setup_seed(args.seed)
    global_transfo2 = transforms.Compose([
        transforms.Resize((1024, 1024)),
    ])
    pil2tensor_transfo = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])
    global_transfo2_subregions = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                            args.data_root + 'seg_rec_mask_1024',
                                                            txtpath=args.data_root + 'chinese_biaozhunwo_1.txt',
                                                            csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                            data_transform=global_transfo2,
                                                            pil2tensor_transform=pil2tensor_transfo,
                                                            data_subregions_transform=global_transfo2_subregions,
                                                            sub_img_size=256, )

    dataset_test = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                            args.data_root + 'seg_rec_mask_1024',
                                                            txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                                            csvpath=args.data_root + 'subregions_label_shanxi_all_woyinying.xlsx',
                                                            data_transform=global_transfo2,
                                                            pil2tensor_transform=pil2tensor_transfo,
                                                            data_subregions_transform=global_transfo2_subregions,
                                                            sub_img_size=256, )

    dataset_test2 = Shanxi_w7masks_5Subregions_wsubroi_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024','/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                                              txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                                              csvpath='/disk3/wjr/dataset/nejm/guizhoudataset/subregions_guizhou_all.xlsx',
                                                              data_transform=global_transfo2,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo2_subregions,
                                                              sub_img_size=256, )
    dataset_test3 = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                            args.data_root + 'seg_rec_mask_1024',
                                                            txtpath=args.data_root + 'chinese_biaozhunwo_1.txt',
                                                            csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                            data_transform=global_transfo2,
                                                            pil2tensor_transform=pil2tensor_transfo,
                                                            data_subregions_transform=global_transfo2_subregions,
                                                            sub_img_size=256, )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_prototype = torch.utils.data.DataLoader(
        dataset_test3,
        batch_size=config.DATA.BATCH_SIZE_SMALL,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test_small = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE_SMALL,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2_small = torch.utils.data.DataLoader(
        dataset_test2,
        batch_size=config.DATA.BATCH_SIZE_SMALL,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    return data_loader_val, data_loader_test, data_loader_test2, data_loader_prototype, data_loader_test_small, data_loader_test2_small

from model_cls import build_model
def save_rng_states():
    # 保存CPU和GPU的RNG状态
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state

def restore_rng_states(cpu_state, cuda_state):
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
from model_cls.pka2net import *
from model_cls.pcam_densenet import *
from model_cls.dmalnet.fusionnet import *
from model_cls.convnext_dinov3 import *
from model_cls.ViP.ViP import ViP
from model_cls.ViP.CoCoOp import CoCoOp
from model_cls.pneullm.mm_adaptation import phellm
from model_cls.dinov3.vit_dinov3 import *
from timm.utils import ModelEma as ModelEma
from model_cls.rad_dino.raddino import rad_dino_gh
from model_cls.vmambav2.vmamba import *
from model_cls.convnext import *
# from model_cls.qwen2_5vl import qwen2_5vision
from torch import optim as optim
def main(config, pt_path, logger_path):
    data_loader_val, data_loader_test, data_loader_test2, data_loader_prototype, data_loader_test_small, data_loader_test2_small = build_phe_loader(
        args)

    logger.info(f"Creating model:{args.model_name}")
    if args.model_name=='resnet50':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        model = get_model('resnet50_8xb32_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        # net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == "slgms":
        model=DINO_VSSM()

    elif args.model_name == "rad_dino":
        model = rad_dino_gh()
        model.head = nn.Linear(768, 2)
    elif args.model_name == "qwen2_5vision":
        model = qwen2_5vision(n_class=2)
    elif args.model_name == "qwen2_5vision_lora":
        model = qwen2_5vision_lora(n_class=2)
    elif args.model_name == "dinov3":
        # model = vit_small(img_size=224,
        #                   in_chans=3,
        #                   pos_embed_rope_base=100,
        #                   pos_embed_rope_normalize_coords="separate",
        #                   pos_embed_rope_rescale_coords=2,
        #                   pos_embed_rope_dtype="fp32",
        #                   qkv_bias=True,
        #                   drop_path_rate=0.0,
        #                   layerscale_init=1.0e-05,
        #                   norm_layer="layernormbf16",
        #                   ffn_layer="mlp",
        #                   ffn_bias=True,
        #                   proj_bias=True,
        #                   n_storage_tokens=4,
        #                   mask_k_bias=True,
        #                   compact_arch_name="vits", )
        # pretrained = '/disk3/wjr/workspace/sec_proj4/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        # state_dict = torch.load(pretrained, map_location='cpu')
        # msg = model.load_state_dict(state_dict, strict=False)
        # print(msg)
        # model.head = nn.Linear(384, 2)

        model = vit_base(img_size=224,
                         patch_size=16,
                         in_chans=3,
                         pos_embed_rope_base=100,
                         pos_embed_rope_normalize_coords="separate",
                         pos_embed_rope_rescale_coords=2,
                         pos_embed_rope_dtype="fp32",
                         qkv_bias=True,
                         drop_path_rate=0.0,
                         layerscale_init=1.0e-05,
                         norm_layer="layernormbf16",
                         ffn_layer="mlp",
                         ffn_bias=True,
                         proj_bias=True,
                         n_storage_tokens=4,
                         mask_k_bias=True,
                         )
        pretrained = '/disk3/wjr/workspace/sec_proj4/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        state_dict = torch.load(pretrained, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model.head = nn.Linear(768, 2)

        # model = vit_large(img_size=224,
        #                  patch_size=16,
        #                  in_chans=3,
        #                  pos_embed_rope_base=100,
        #                  pos_embed_rope_normalize_coords="separate",
        #                  pos_embed_rope_rescale_coords=2,
        #                  pos_embed_rope_dtype="fp32",
        #                  qkv_bias=True,
        #                  drop_path_rate=0.0,
        #                  layerscale_init=1.0e-05,
        #                  norm_layer="layernormbf16",
        #                  ffn_layer="mlp",
        #                  ffn_bias=True,
        #                  proj_bias=True,
        #                  n_storage_tokens=4,
        #                  mask_k_bias=True,
        #                  untie_global_and_local_cls_norm=False,
        #                  )
        # pretrained = '/disk3/wjr/workspace/sec_proj4/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        # state_dict = torch.load(pretrained, map_location='cpu')
        # msg = model.load_state_dict(state_dict, strict=False)
        # print(msg)
        # model.head = nn.Linear(1024, 2)

    elif args.model_name == 'ViP':
        model=ViP(device=device, backbone_name="RN50x64")
    elif args.model_name == 'CoCoOp':
        model = CoCoOp(device=device)
        # model=CoCoOp(device=device, backbone_name="RN50x64")
    elif args.model_name == 'pneullm':
        model=phellm()
    elif args.model_name == 'hrnet18':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w18_3rdparty_8xb32-ssld_in1k_20220120-455f69ea.pth'
        model = get_model('hrnet-w18_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == 'hrnet48':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth'
        model = get_model('hrnet-w48_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(2048, 2)

    elif args.model_name == "convnext":
        model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
        model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_org(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "convnext_dinov3":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_dino3(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
        model = convnextbase_dino3(pretrained=True, drop_path_rate=0.)
        model.head.fc = nn.Linear(1024, 2)
        # model = convnextlarge_dino3(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "dinov2":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'
        model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=pretrained_cfg)
        model.backbone.ln1 = nn.Linear(384, 2, bias=True)
    elif args.model_name == 'vim':
        model = build_model(config)
        model.head = nn.Linear(192, 2)
    elif args.model_name == 'PKA2_Net':
        model = PKA2_Net(n_class=2)
    elif args.model_name == 'pcam':
        model = pcam_dense121(num_classes=[1, 1])
        model.to(device)
    elif args.model_name == "swintiny":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(768, 2, bias=True)
    elif args.model_name == "vit-base":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
        model = get_model('vit-base-p16_in21k-pre_3rdparty_in1k-384px', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.layers.head = nn.Linear(768, 2, bias=True)
    elif args.model_name == "vit-large":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth'
        model = get_model('vit-large-p16_in21k-pre_3rdparty_in1k-384px', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.layers.head = nn.Linear(1024, 2, bias=True)
    elif args.model_name == 'pvt':
        model = pvt_tiny(pretrained=True)
        model.head = nn.Linear(512, 2)
    elif args.model_name == 'dmalnet':
        args.output_dir = '/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/DMALNet/_lr0.0003/models_params/RR_avgpool/'
        model = fusion_net(pretrained_pth=args.output_dir)
    model.to(device)
    ckp_path = pt_path
    if os.path.isfile(ckp_path):
        if args.model_name == "pneullm":
            checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)['model']
            model_dict = model.state_dict()
            # 关键在于下面这句，从model_dict中读取key、value时，用if筛选掉不需要的网络层
            pretrained_dict = {key: value for key, value in checkpoint.items() if
                               (key in model_dict)}
            # 将参数加载到模型中
            msg = model.load_state_dict(pretrained_dict, strict=False)


        elif args.model_name == "slgms":
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['teacher']
            state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
            msg = model.load_state_dict(state_dict, strict=False)

        # model=train_one_epoch(config, model, data_loader_val)
        # validate_network(data_loader_val, model, val_index='shanxi_val')
        # if args.model_name == "slgms":
        #     state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['teacher']
        #     state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        #     msg = model.load_state_dict(state_dict, strict=False)
        # else:
        #     state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
        #     msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"TTA ON SHANXITEST..........................................")
        model = train_one_epoch(config, model, data_loader_prototype, data_loader_test, data_loader_test_small)
        shanxi_test_shot = validate_network_glb(data_loader_test, model, 'shanxi_test')
        save_state = {'model': model.state_dict()}
        save_path = os.path.join(logger_path, 'shanxi_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)

        gc.collect()
        if args.model_name == "slgms":
            model = DINO_VSSM()
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['teacher']
            state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
            model.head.fc = nn.Linear(768, 2)
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
            msg = model.load_state_dict(state_dict, strict=False)

        model.to(device)
        shanxi_test_org = validate_network_glb(data_loader_test, model, 'shanxi_test')
        guizhou_test_org = validate_network_glb(data_loader_test2, model, 'guizhou_test')
        logger.info(f"TTA ON GUIZHOUTEST..........................................")
        model = train_one_epoch(config, model, data_loader_prototype, data_loader_test2, data_loader_test2_small)
        guizhoutest_shot = validate_network_glb(data_loader_test2, model, 'guizhou_test')
        save_state = {'model': model.state_dict()}
        save_path = os.path.join(logger_path,'guizhou_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        gc.collect()
        # obtain_label(data_loader_val, model,'shanxival')
        # obtain_label(data_loader_test, model, 'shanxitest')
        # obtain_label(data_loader_test2, model, 'guizhoutest')
        return shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot


import operator




from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.5, axis_labels=None, valdata='shanxi_test'):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm0 = confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm0.astype('float') / cm0.sum(axis=1)[:, np.newaxis]  # 归一化

    class_names = ['Normal', 'Sick']  # 类名称
    # 设置字体大小
    title_font_size = 26  # 标题字体大小
    label_font_size = 24  # 坐标轴标签字体大小
    tick_font_size = 22  # 刻度字体大小
    text_font_size = 22  # 矩阵内文本字体大小
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'), vmin=0, vmax=1)
    # 添加 colorbar
    cbar = plt.colorbar()

    # 设置 colorbar 的字体大小
    cbar.ax.tick_params(labelsize=22)  # 设置 colorbar 刻度字体大小
    # cbar.set_label('Color Scale', fontsize=14)  # 设置 colorbar 标签字体大小

    # plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title, fontsize=title_font_size)

    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = class_names
    plt.xticks(num_local, axis_labels, rotation=0, fontsize=tick_font_size)  # 将类名称印在x轴坐标上，并倾斜45度
    # plt.yticks(num_local, axis_labels, rotation=90, fontsize=tick_font_size)  # 将类名称印在y轴坐标上
    plt.yticks(num_local, axis_labels, fontsize=tick_font_size)
    labels = plt.gca().get_yticklabels()  # 获取 y 轴标签
    plt.setp(labels, rotation=90, va='center')  # 调整对齐方式
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                count = int(cm[i][j] * cm.sum(axis=1)[i])  # 样本数量
                plt.text(j, i - 0.08, f'{int(cm0[i][j])}',  # 显示样本数量
                         ha="center", va="center", fontsize=text_font_size,
                         color="white" if cm[i][j] > thresh else "black")

                plt.text(j, i + 0.08, format(float(cm[i][j] * 100), '.2f') + '%',
                         ha="center", va="center", fontsize=text_font_size,
                         color="white" if cm[i][j] > thresh else "black")

    # 设置保存路径
    # save_path = config.OUTPUT + '/'+valdata+'_confusion_map.png'
    # if save_path:
    #     plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距

def adaptive_calculate_metrics(all_outputs, all_targets_onehot,threshold=None, valdata='shanxi_test',num_classes=2, plot_roc=True, save_fpr_tpr=True):
    # # 将输出通过 Sigmoid 激活函数转换为概率
    probabilities=all_outputs
    all_targets=all_targets_onehot
    # 初始化存储各项指标的列表
    auc_scores = []
    optimal_thresholds = []
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    sensitivity_scores = []
    specificity_scores = []

    for i in range(1):
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(all_targets[:, i], probabilities[:, i])
        if threshold == None:
            # mccs, threshold = compute_mccs(all_targets, probabilities)
            # print(threshold)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
            # print(optimal_threshold)

        auc = roc_auc_score(all_targets[:, i], probabilities[:, i])
        if plot_roc:
            # plt.figure(figsize=(8, 6))
            # plt.plot(fpr, tpr, label=f'AUC = {auc * 100:.2f}' + '%')
            # plt.plot([0, 1], [0, 1], 'k--')
            # 找到最接近特定 threshold 的索引
            idx = np.argmin(np.abs(thresholds - threshold))
            # 获取对应的 fpr 和 tpr 值
            specific_fpr = fpr[idx]
            specific_tpr = tpr[idx]
            # 使用 plt.scatter 标记特定点
            # plt.scatter(specific_fpr, specific_tpr, marker='*', color='red', s=200, zorder=10)
            #
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # # plt.legend(loc='lower right')
            # plt.legend(loc='lower right', prop={'size': 20})
            # save_path = config.OUTPUT + '/' + valdata + '_roc_map.pdf'
            # plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
            # # # 显示
            # # plt.show()
            # # 自动关闭图像
            # plt.close()

        auc_scores.append(auc)

        # 使用最优阈值计算预测值
        y_pred = (probabilities[:, i] >= threshold).astype(int)
        # y_pred = (probabilities[:, i] >= 0.5).astype(int)
        # 计算各项指标
        acc_scores.append(accuracy_score(all_targets[:, i], y_pred))
        f1_scores.append(f1_score(all_targets[:, i], y_pred))
        precision_scores.append(precision_score(all_targets[:, i], y_pred))
        recall_scores.append(recall_score(all_targets[:, i], y_pred))

        # 1. 计算混淆矩阵
        plot_matrix(all_targets[:, i], y_pred, [0, 1], title=None, thresh=0.6, axis_labels=None, valdata=valdata)

        # 计算 Sensitivity 和 Specificity
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], y_pred).ravel()
        sensitivity_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if save_fpr_tpr:
        # 在 fpr 最后一个数后接上 specific_fpr
        fpr = np.append(fpr, specific_fpr)
        tpr = np.append(tpr, specific_tpr)
        # np.save(config.OUTPUT + '/' + valdata + '_fpr.npy', fpr)
        # np.save(config.OUTPUT + '/' + valdata + '_tpr.npy', tpr)
    # 计算平均值
    auc_scores = np.array(auc_scores)
    auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])  # 计算有效 AUC 的平均值
    acc = np.mean(acc_scores)
    f1 = np.mean(f1_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    sensitivity = np.mean(sensitivity_scores)
    specificity = np.mean(specificity_scores)

    return acc * 100, auc_scores * 100, sensitivity * 100, specificity * 100, f1 * 100, precision * 100, recall * 100, threshold

def local2global_prob(local_probs):
    # local_probs = torch.nn.functional.softmax(local_logits, dim=-1)  # [B, 6, 2]
    p0 = local_probs[..., 1]  # [B, 6]：每个肺区0级的概率

    # 2. 计算每个肺区≥1级的概率 s = 1 - p0
    s = 1.0 - p0  # [B, 6]

    # 3. 计算 P(满足数=0)：所有6个肺区都是0级的概率
    prod_p0 = torch.prod(p0, dim=1)  # [B]，沿肺区维度（6）求积

    # 4. 计算 P(满足数=1)：恰好1个肺区≥1级，其余5个为0级的概率
    # 公式：sum( s_i * (prod_p0 / p0_i) )，处理p0_i=0的情况（加eps）
    inv_p0 = 1.0 / (p0 + 1e-8)  # [B, 6]，避免除以0
    term = s * inv_p0 * prod_p0.unsqueeze(1)  # [B, 6]，prod_p0.unsqueeze(1)扩展为[B,1]
    P_k1 = torch.sum(term, dim=1)  # [B]，沿肺区维度求和

    # 5. 规则违反概率：0个或1个肺区满足1级
    V_prob = prod_p0 + P_k1  # [B]
    global_output=torch.cat((1-V_prob.unsqueeze(1), V_prob.unsqueeze(1)), dim=1)
    return global_output

@torch.no_grad()
def validate_network_glb(val_loader, model, valdata='shanxi_val'):
    model.eval()
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    data_results = []
    # 初始化输出和目标列表
    all_outputs = []
    all_local_outputs = []
    all_local_global_outputs = []
    all_targets = []
    total_loss=0
    id=0
    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(val_loader):
        samples = images[0].to(device)
        for ii in range(len(labels)):
            labels[ii] = labels[ii].to(device)
        targets = labels[0].to(device)
        if isinstance(masks[0], list):
            sample_masks = masks[0]
            for ii in range(len(sample_masks)):
                sample_masks[ii] = sample_masks[ii].to(device)
        else:
            sample_masks = masks[0].to(device)
        subregions_imgs = []
        subregions_masks = []
        subregions_labels = []
        # subregions_label = labels[1]
        index = 0
        for ii in range(labels[1].shape[0]):
            index = 1
            sub_6imgs = []
            sub_6masks = []
            sub_6labels = []
            for jj in range(6):
                subimg = images[jj + 1]
                submask = masks[jj + 1]
                sublabel = labels[jj + 1]
                sub_6imgs.append(subimg[ii, ...].unsqueeze(0))
                sub_6masks.append(submask[ii, ...].unsqueeze(0))
                sub_6labels.append(sublabel[ii, ...].unsqueeze(0))
            sub_6imgs = torch.cat(sub_6imgs, dim=0).to(device)
            sub_6masks = torch.cat(sub_6masks, dim=0).to(device)
            # sub_6labels = torch.cat(sub_6labels, dim=0).to(device)
            subregions_imgs.append(sub_6imgs)
            subregions_masks.append(sub_6masks)
            # subregions_labels.append(sub_6labels)
        if index != 0:
            subregions_imgs = torch.cat(subregions_imgs, dim=0).to(device)
            subregions_masks = torch.cat(subregions_masks, dim=0).to(device)

        all_rois = []
        sub_rois = torch.stack(sub_rois)
        for bb in range(sub_rois.shape[1]):
            for kk in range(sub_rois.shape[0]):
                sub_bb_roi = sub_rois[kk, bb, :]
                roi_with_batch_idx = torch.cat([torch.tensor([bb]).float(), sub_bb_roi])
                all_rois.append(roi_with_batch_idx)

        samples = (samples * sample_masks[0]).to(device)
        pred, _ = model(samples)
        if args.model_name == "dinov2":
            pred = pred[1]

        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

        all_targets.extend(targets.cpu().numpy())

        subregions_outputs, _ = model(subregions_imgs * subregions_masks)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
        subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
        global_prob = local2global_prob(subregions_outputs2)
        all_local_outputs.extend(global_prob.data.cpu().numpy())
        all_local_global_outputs.extend(((global_prob + prob) / 2).data.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_local_outputs = np.array(all_local_outputs)
    all_local_global_outputs = np.array(all_local_global_outputs)
    all_targets = np.array(all_targets)
    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg = (acc + auc + specificity + sensitivity) / 4
    logger.info(
        f'Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)

    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics(
        all_local_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    logger.info(
        f'Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)

    acc_local_global, auc_local_global, sensitivity_local_global, specificity_local_global, f1_local_global, precision_local_global, recall_local_global, threshold_mcc = adaptive_calculate_metrics(
        all_local_global_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local_global = (acc_local_global + auc_local_global + specificity_local_global + sensitivity_local_global) / 4
    logger.info(
        f'Local2_global_results Average: ' + "%.4f" % avg_local_global + ' Accuracy: ' + "%.4f" % acc_local_global + ' AUC.5: ' + "%.4f" % auc_local_global + ' Sensitivity: ' + "%.4f" % sensitivity_local_global + ' Specificity.5: ' + "%.4f" % specificity_local_global)

    return [avg, acc, sensitivity, specificity, auc],[avg_local , acc_local , sensitivity_local , specificity_local , auc_local ], [avg_local_global, acc_local_global, sensitivity_local_global, specificity_local_global, auc_local_global],
    # return loss_meter.avg, auc, acc, writer
@torch.no_grad()
def validate_network(val_loader, model, val_index='shanxi_val'):
    model.eval()
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    data_results = []
    total_loss=0
    id=0
    for i, (inp, mask, target, imgname) in enumerate(val_loader):
        result = {}
        result["Image Index"] = imgname[0]
        result["Finding Labels"] = imgname[0].split('_')[0]
        id+=1
        # move to gpu
        inp = (inp* mask).to(device)
        # inp = inp.to(device)
        # inp = inp.cuda(non_blocking=True)
        target = target.to(device)

        # forward
        with torch.no_grad():
            output = model(inp)
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            output=output[0]
        prob = torch.nn.functional.softmax(output, dim=1).float()
        for bb in range(prob.shape[0]):
            aa = torch.argmax(output[bb,:])
            if aa == 0:
                pred_global_labels = 'Sick'
            else:
                pred_global_labels = 'Health'
            result["Pred Labels"] = pred_global_labels
            result["Pred logits"] = '%.4f' % prob[bb, aa].cpu().detach().numpy()
            data_results.append(result)
            # 收集输出和目标
            all_outputs.extend(prob[bb,:].unsqueeze(0).data.cpu().numpy())
            # targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

            all_targets.extend(target[bb,:].unsqueeze(0).cpu().numpy())

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    # df_data_results = pd.DataFrame(data_results)
    # Path(config.OUTPUT + '/' + 'analyze').mkdir(parents=True, exist_ok=True)
    # df_data_results.to_csv(config.OUTPUT + '/' + val_index + '_dataprob_result.csv', index=False)

    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=val_index,
        plot_roc=True,
        save_fpr_tpr=True)
    avg = (acc + auc + specificity + sensitivity) / 4
    print(f'{val_index} Sensitivity: {sensitivity}  Specificity: {specificity}, Accuracy: {acc}  AUC: {auc}, avg: {avg} ')
    # logger.info(
    #     f'lr: {args.base_lr} {valdata} {ptnames}' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)
    return avg, acc, sensitivity, specificity, auc
    # return loss_meter.avg, auc, acc, writer
def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def compute_structure_fisher_scores_pesudo(model, support_loader, mem_label, device):
    """
    Compute Structure Fisher Scores for all convolutional layers in the model.

    Args:
        model (nn.Module): Pre-trained model (e.g., ResNet)
        support_loader (DataLoader): DataLoader for support set (few-shot labeled data)
        device (torch.device): e.g., 'cuda' or 'cpu'

    Returns:
        dict: {layer_name: structure_fisher_score}
    """
    model.eval()
    model.to(device)

    # Step 1: Initialize storage for squared gradients per parameter
    fisher_info = {}
    layer_names = []

    # Register hooks to capture gradients of conv weights
    handles = []

    def make_hook(name):
        def hook(grad):
            # Accumulate squared gradients: (1/|K|) * sum( (dL/dθ)^2 )
            if name not in fisher_info:
                fisher_info[name] = torch.zeros_like(grad).to(device)
            fisher_info[name] += grad.pow(2)

        return hook

    # Attach hooks to all Conv2d weight parameters

    for name, param in model.named_parameters():
        if 'conv' in name and param.requires_grad:
            layer_names.append(name)
            handle = param.register_hook(make_hook(name))
            handles.append(handle)
    # for name, param in model.named_parameters():
    #     layer_names.append(name)
    #     handle = param.register_hook(make_hook(name))
    #     handles.append(handle)

    total_samples = 0

    # Step 2: Forward + backward on support set
    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(support_loader):
        samples = images[0].to(device)
        for ii in range(len(labels)):
            labels[ii] = labels[ii].to(device)
        targets = labels[0].to(device)
        targetss = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
        if isinstance(masks[0], list):
            sample_masks = masks[0]
            for ii in range(len(sample_masks)):
                sample_masks[ii] = sample_masks[ii].to(device)
        else:
            sample_masks = masks[0].to(device)
        inp = (samples * sample_masks[0]).to(device)

        pred = mem_label[idx * config.DATA.BATCH_SIZE_SMALL:idx * config.DATA.BATCH_SIZE_SMALL + inp.shape[0]]
        batch_size = inp.size(0)
        total_samples += batch_size

        model.zero_grad()
        logits, _ = model(inp)
        loss = nn.CrossEntropyLoss()(logits, pred)
        loss.backward()  # This triggers the hooks

    # Remove hooks
    for h in handles:
        h.remove()

    # Step 3: Average by number of samples → Empirical Fisher Info per param
    for name in fisher_info:
        fisher_info[name] /= total_samples  # F_θ = (1/|K|) Σ (∂log p / ∂θ)^2

    # Step 4: Compute Structure Fisher Score per layer (max over params in layer)
    structure_fisher = {}

    # Group by layer (assuming param names like 'layer1.0.conv1.weight')
    layer_groups = {}
    for name in fisher_info:
        # Extract layer identifier (e.g., 'layer1.0.conv1')
        layer_key = '.'.join(name.split('.')[:-1])  # remove '.weight'
        if layer_key not in layer_groups:
            layer_groups[layer_key] = []
        layer_groups[layer_key].append(fisher_info[name])
    for layer_key, tensors in layer_groups.items():
        # More robust: use mean of max per tensor
        max_vals = torch.stack([t.max() for t in tensors])
        structure_fisher[layer_key] = max_vals.mean().item()  # or .median()
    # for layer_key, param_fishers in layer_groups.items():
    #     # Take max over all parameters in this layer
    #     max_fisher = torch.stack([p.max() for p in param_fishers]).max().item()
    #     structure_fisher[layer_key] = max_fisher

    return structure_fisher

def train_one_epoch(config, model, data_loader_prototype, data_loader, dataloader_small):
    # Step 1: 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    target_params = total_params * 0.15  # 前15%的参数量

    # Step 2: 按顺序收集每层的参数量（去重 layer_key）
    layer_param_count = {}
    layer_order = []  # 保持层的出现顺序
    seen_layers = set()

    for name, param in model.named_parameters():
        layer_key = '.'.join(name.split('.')[:-1])
        if layer_key not in seen_layers:
            layer_order.append(layer_key)
            seen_layers.add(layer_key)
        # 累计该层的参数量（一个 layer_key 可能有多个 param，如 weight + bias）
        layer_param_count[layer_key] = layer_param_count.get(layer_key, 0) + param.numel()
    # Step 3: 从浅层开始累加，直到 >= target_params
    cumulative = 0
    trainable_layer_set = set()
    for layer_key in layer_order:
        cumulative += layer_param_count[layer_key]
        trainable_layer_set.add(layer_key)
        if cumulative >= target_params:
            break

    # Step 4: 设置 requires_grad
    trainable_names = []
    for name, param in model.named_parameters():
        layer_key = '.'.join(name.split('.')[:-1])
        if layer_key in trainable_layer_set:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
    print('Trainable parameters:', trainable_names)
    logger.info('Trainable parameters:', trainable_names)

    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    optimizer = build_optimizer(config, model)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader))


    logger.info("Start training")

    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    for epoch in range(config.TRAIN.EPOCHS):
        model.eval()
        # validate_network_glb(data_loader, model, 'guizhou_test')
        # mem_label, imgnames = obtain_label(data_loader, model,'1prototype')
        # mem_label, imgnames = obtain_multi_prototype_label(data_loader, model,
        #                                                                  NUM_PROTO=config.NUM_PROTOTYPES,
        #                                                                  valdata=f'{config.NUM_PROTOTYPES}prototypes')

        mem_label, reliable_mask, imgnames = obtain_multi_centers_multi_prototype_label(data_loader_prototype, data_loader, model, NUM_PROTO = config.NUM_PROTOTYPES, valdata = f'{config.NUM_PROTOTYPES}prototypes_2centers')
        mem_label = torch.from_numpy(mem_label).to(device)
        # mem_sub_label = torch.from_numpy(mem_sub_label).to(device)
        model.train()
        loss_meter = AverageMeter()
        loss_glb_cls_meter = AverageMeter()
        loss_loc_cls_meter = AverageMeter()
        loss_con_meter = AverageMeter()
        loss_con_fea_meter = AverageMeter()
        loss_ent_cls_meter = AverageMeter()
        for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(data_loader):

            samples = images[0].to(device)
            for ii in range(len(labels)):
                labels[ii] = labels[ii].to(device)
            targets = labels[0].to(device)
            targetss = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
            if isinstance(masks[0], list):
                sample_masks = masks[0]
                for ii in range(len(sample_masks)):
                    sample_masks[ii] = sample_masks[ii].to(device)
            else:
                sample_masks = masks[0].to(device)
            subregions_imgs = []
            subregions_masks = []
            subregions_labels = []
            # subregions_label = labels[1]
            index = 0
            for ii in range(labels[1].shape[0]):
                index = 1
                sub_6imgs = []
                sub_6masks = []
                sub_6labels = []
                for jj in range(6):
                    subimg = images[jj + 1]
                    submask = masks[jj + 1]
                    sublabel = labels[jj + 1]
                    sub_6imgs.append(subimg[ii, ...].unsqueeze(0))
                    sub_6masks.append(submask[ii, ...].unsqueeze(0))
                    sub_6labels.append(sublabel[ii, ...].unsqueeze(0))
                sub_6imgs = torch.cat(sub_6imgs, dim=0).to(device)
                sub_6masks = torch.cat(sub_6masks, dim=0).to(device)
                # sub_6labels = torch.cat(sub_6labels, dim=0).to(device)
                subregions_imgs.append(sub_6imgs)
                subregions_masks.append(sub_6masks)
                # subregions_labels.append(sub_6labels)
            if index != 0:
                subregions_imgs = torch.cat(subregions_imgs, dim=0).to(device)
                subregions_masks = torch.cat(subregions_masks, dim=0).to(device)

            images = (samples * sample_masks[0]).to(device)
            pred = mem_label[idx * config.DATA.BATCH_SIZE:idx * config.DATA.BATCH_SIZE + images.shape[0]]

            outputs_test,glb_feas = model(images)
            subregions_outputs, sub_feas = model(subregions_imgs * subregions_masks)
            subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
            subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
            sub_feas2 = sub_feas.view(samples.shape[0], 6, -1)
            glb_fea=F.softmax(glb_feas, dim=-1)
            local_global_losses=[]
            for ff in range(6):
                fea_lossii = torch.sum(-glb_fea.detach() * F.log_softmax(sub_feas2[:,ff,...], dim=-1), dim=-1)
                local_global_losses.append(fea_lossii.unsqueeze(0))
            local_global_losses = torch.cat(local_global_losses, dim=0)
            score_mean_global = pred.detach()

            fea_loss_local =local_global_losses.permute(1,0).flatten().mean()
            score_mean_global_expanded = score_mean_global.unsqueeze(1).repeat(1, 6).flatten()  # (N, 6)
            cls_loss_local = nn.CrossEntropyLoss()(subregions_outputs, score_mean_global_expanded)

            V_prob = local2global_prob(subregions_outputs2)
            # rule_loss = nn.CrossEntropyLoss()(outputs_test, V_prob)
            outputs_test2=torch.nn.functional.softmax(outputs_test, dim=1)
            p = outputs_test2.clamp(min=1e-6)
            q = V_prob.clamp(min=1e-6)
            m = 0.5 * (p + q)

            rule_loss = 0.5 * F.kl_div(p.log(), m, reduction='sum') + \
                        0.5 * F.kl_div(q.log(), m, reduction='sum')

            # rule_loss = 0.5*F.kl_div((outputs_test2 + 1e-8).log(), V_prob.detach(), reduction='sum')+0.5*F.kl_div((V_prob+ 1e-8).log(), outputs_test2.detach(), reduction='sum')

            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            loss = classifier_loss+config.LOCRADIO*rule_loss
            # loss = classifier_loss * 0.3 + entropy_loss * 1.
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            loss_glb_cls_meter.update(classifier_loss.item(), images.size(0))
            loss_loc_cls_meter.update(cls_loss_local.item(), images.size(0))
            loss_con_meter.update(rule_loss.item(), images.size(0))
            loss_con_fea_meter.update(fea_loss_local.item(), images.size(0))
            loss_ent_cls_meter.update(entropy_loss.item(), images.size(0))
            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'glb classi_loss {loss_glb_cls_meter.val:.4f} ({loss_glb_cls_meter.avg:.4f})\t'
                    f'loc classi_loss {loss_loc_cls_meter.val:.4f} ({loss_loc_cls_meter.avg:.4f})\t'
                    f'rule_loss {loss_con_meter.val:.8f} ({loss_con_meter.avg:.8f})\t'
                    f'fea_con_loss {loss_con_fea_meter.val:.4f} ({loss_con_fea_meter.avg:.4f})\t'
                    f'entropy_loss {loss_ent_cls_meter.val:.4f} ({loss_ent_cls_meter.avg:.4f})\t'
                )
    # mem_label = obtain_label(data_loader, model)
    mem_label, _, imgnames = obtain_multi_centers_multi_prototype_label(data_loader_prototype, data_loader, model, NUM_PROTO = config.NUM_PROTOTYPES)
    return model





def merge_dict_list(dict_list):
    merged = []
    for item in dict_list:
        for k, v in item.items():
            merged.append(v)
    # 拼接每个 key 对应的所有 tensor
    return torch.cat(merged, dim=0)
def merge_list2dict(predict, dict_list):
    merged = []
    ii=0
    for item in dict_list:
        for k, v in item.items():
            merged.append([{k: predict[ii:ii+v.shape[0],...]}])
            ii+=v.shape[0]
    # 拼接每个 key 对应的所有 tensor
    return merged
from scipy.spatial.distance import cdist
from sklearn.isotonic import IsotonicRegression
@torch.no_grad()
def obtain_isotonic_label(loader, model, valdata=''):
    model.eval()
    start_test = True
    img_names = []
    with torch.no_grad():
        for idx, (images, masks, targetss, imaname) in enumerate(loader):
            for iimgname in imaname:
                img_names.append(iimgname)
            images = (images * masks).to(device)
            outputs, feas = model(images)
            if start_test:
                all_fea = [{f'{idx}': feas.float().cpu()}]
                all_output = [{f'{idx}': outputs.float().cpu()}]
                all_label = [{f'{idx}': targetss[:, 1].float().cpu()}]
                start_test = False
            else:
                all_fea.append({f'{idx}': feas.float().cpu()})
                all_output.append({f'{idx}': outputs.float().cpu()})
                all_label.append({f'{idx}': targetss[:, 1].float().cpu()})

    # 分别合并 fea、output、label
    final_fea = merge_dict_list(all_fea)
    final_output = merge_dict_list(all_output)
    final_output = nn.Softmax(dim=1)(final_output)
    final_label = merge_dict_list(all_label)

    _, predict = torch.max(final_output, 1)
    final_fea = (final_fea.t() / torch.norm(final_fea, p=2, dim=1)).t()
    accuracy = torch.sum(torch.squeeze(predict).float() == final_label).item() / float(final_label.size()[0])

    final_fea = final_fea.float().cpu().numpy()
    K = final_output.size(1)
    aff = final_output.float().cpu().numpy()
    initc = aff.transpose().dot(final_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    subcls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(subcls_count > 0)
    labelset = labelset[0]
    # print(labelset)
    dd = cdist(final_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(final_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(final_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == final_label.float().numpy()) / len(final_fea)
    log_str = valdata + ' Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str + '\n')

    iso = IsotonicRegression(out_of_bounds='clip').fit(
        np.array(final_output[:, 1]), np.array(pred_label)
    )
    all_output_pseudo = torch.tensor(iso.predict(np.array(final_output[:, 1])))
    all_output_pseudo=torch.cat((1-all_output_pseudo.unsqueeze(-1), all_output_pseudo.unsqueeze(-1)),dim=-1)
    _, predict = torch.max(all_output_pseudo, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == final_label).item() / float(final_label.size()[0])

    K = all_output_pseudo.size(1)
    aff = all_output_pseudo.float().cpu().numpy()
    initc = aff.transpose().dot(final_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]
    # print(labelset)
    dd = cdist(final_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(final_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(final_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == final_label.float().numpy()) / len(final_fea)
    log_str =  valdata + ' pseudo isotonic Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str + '\n')
    model.train()
    return pred_label.astype('int'), img_names #, labelset
@torch.no_grad()
def obtain_label(loader, model, valdata=''):
    model.eval()
    start_test = True
    img_names=[]
    with torch.no_grad():
        for idx, (images,masks,targetss,imaname) in enumerate(loader):
            for iimgname in imaname:
                img_names.append(iimgname)
            images = (images * masks).to(device)
            outputs, feas = model(images)
            if start_test:
                all_fea = [{f'{idx}': feas.float().cpu()}]
                all_output = [{f'{idx}': outputs.float().cpu()}]
                all_label = [{f'{idx}': targetss[:,1].float().cpu()}]
                start_test = False
            else:
                all_fea.append({f'{idx}': feas.float().cpu()})
                all_output.append({f'{idx}': outputs.float().cpu()})
                all_label.append({f'{idx}': targetss[:,1].float().cpu()})

    # 分别合并 fea、output、label
    final_fea = merge_dict_list(all_fea)
    final_output = merge_dict_list(all_output)
    final_output = nn.Softmax(dim=1)(final_output)
    final_label = merge_dict_list(all_label)

    _, predict = torch.max(final_output, 1)
    final_fea = (final_fea.t() / torch.norm(final_fea, p=2, dim=1)).t()
    accuracy = torch.sum(torch.squeeze(predict).float() == final_label).item() / float(final_label.size()[0])

    final_fea = final_fea.float().cpu().numpy()
    K = final_output.size(1)
    aff = final_output.float().cpu().numpy()
    initc = aff.transpose().dot(final_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    subcls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(subcls_count > 0)
    labelset = labelset[0]
    # print(labelset)
    dd = cdist(final_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(final_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(final_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == final_label.float().numpy()) / len(final_fea)
    log_str = valdata+' Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')
    model.train()
    return pred_label, img_names #, labelset
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
@torch.no_grad()
def obtain_multi_prototype_label(loader, model, NUM_PROTO = 3, INIT_WEIGHT = 0.5, valdata = ''):
    #NUM_PROTO = 3  # 每个类别的 prototype 数（2~3 推荐）
    #INIT_WEIGHT = 0.5 softmax-weighted center 与 kmeans center 融合比例
    model.eval()
    start_test = True
    img_names = []

    with torch.no_grad():
        for idx, (images, masks, targetss, imaname) in enumerate(loader):
            for name in imaname:
                img_names.append(name)

            images = (images * masks).to(device)
            outputs, feas = model(images)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = targetss[:, 1].float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, targetss[:, 1].float().cpu()), 0)

    # -----------------------------
    # Softmax prediction
    # -----------------------------
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(
        predict.float() == all_label
    ).item() / float(all_label.size(0))

    # -----------------------------
    # Feature normalization
    # -----------------------------
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    final_fea = all_fea.float().cpu().numpy()
    final_output = all_output.float().cpu().numpy()
    final_label = all_label.float().cpu().numpy()
    predict = predict.cpu().numpy()

    K = final_output.shape[1]

    # -----------------------------
    # Define labelset (important!)
    # -----------------------------
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)[0]

    # =====================================================
    # Softmax-weighted + Per-class K-means Prototypes
    # =====================================================



    # ---------- Step 1: softmax-weighted single prototype ----------
    aff = final_output
    initc_soft = aff.T @ final_fea
    initc_soft = initc_soft / (1e-8 + aff.sum(axis=0)[:, None])

    # ---------- Step 2: per-class K-means prototypes ----------
    all_protos = []
    proto_labels = []

    for cls in labelset:
        idx = np.where(predict == cls)[0]
        feats_cls = final_fea[idx]

        if len(feats_cls) == 0:
            continue

        if len(feats_cls) <= NUM_PROTO:
            centers = feats_cls
        else:
            kmeans = KMeans(
                n_clusters=NUM_PROTO,
                random_state=0,
                n_init='auto'
            ).fit(feats_cls)
            centers = kmeans.cluster_centers_

        # 融合 softmax prototype（SHOT 原味）
        for c in centers:
            # fused_center = INIT_WEIGHT * initc_soft[cls] + (1 - INIT_WEIGHT) * c
            # all_protos.append(fused_center)
            all_protos.append(c)
            proto_labels.append(cls)

    initc = np.vstack(all_protos)
    proto_labels = np.array(proto_labels)

    # ---------- Step 3: nearest prototype assignment ----------
    dd = cdist(final_fea, initc, 'cosine')
    nearest_proto = dd.argmin(axis=1)
    pred_label = proto_labels[nearest_proto]

    # ---------- Step 4: SHOT-style refinement (1 round) ----------
    for _ in range(1):
        all_protos = []
        proto_labels = []

        for cls in labelset:
            idx = np.where(pred_label == cls)[0]
            feats_cls = final_fea[idx]

            if len(feats_cls) == 0:
                continue

            if len(feats_cls) <= NUM_PROTO:
                centers = feats_cls
            else:
                kmeans = KMeans(
                    n_clusters=NUM_PROTO,
                    random_state=0,
                    n_init='auto'
                ).fit(feats_cls)
                centers = kmeans.cluster_centers_

            for c in centers:
                all_protos.append(c)
                proto_labels.append(cls)

        initc = np.vstack(all_protos)
        proto_labels = np.array(proto_labels)

        dd = cdist(final_fea, initc, 'cosine')
        nearest_proto = dd.argmin(axis=1)
        pred_label = proto_labels[nearest_proto]

    # -----------------------------
    # Final accuracy
    # -----------------------------
    acc = np.sum(pred_label == final_label) / len(final_fea)
    log_str = valdata + ' Accuracy = {:.2f}% -> {:.2f}%'.format(
        accuracy * 100, acc * 100
    )
    print(log_str + '\n')

    model.train()
    return pred_label.astype(int), img_names
@torch.no_grad()
def obtain_multi_centers_multi_prototype_label(loader_prototype, loader, model, NUM_PROTO = 3, INIT_WEIGHT = 0.5, valdata = ''):
    #NUM_PROTO = 3  # 每个类别的 prototype 数（2~3 推荐）
    #INIT_WEIGHT = 0.5 softmax-weighted center 与 kmeans center 融合比例
    model.eval()
    start_test = True
    img_names = []
    with torch.no_grad():
        for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(loader_prototype):
            samples = images[0].to(device)
            for ii in range(len(labels)):
                labels[ii] = labels[ii].to(device)
            targets = labels[0].to(device)
            targetss = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
            if isinstance(masks[0], list):
                sample_masks = masks[0]
                for ii in range(len(sample_masks)):
                    sample_masks[ii] = sample_masks[ii].to(device)
            else:
                sample_masks = masks[0].to(device)
            images = (samples * sample_masks[0]).to(device)
            outputs, feas = model(images)
            if start_test:
                all_prototype_fea = feas.float().cpu()
                all_prototype_output = outputs.float().cpu()
                all_prototype_label = targetss[:, 1].float().cpu()
                start_test = False
            else:
                all_prototype_fea = torch.cat((all_prototype_fea, feas.float().cpu()), 0)
                all_prototype_output = torch.cat((all_prototype_output, outputs.float().cpu()), 0)
                all_prototype_label = torch.cat((all_prototype_label, targetss[:, 1].float().cpu()), 0)
    all_prototype_fea = (all_prototype_fea.t() / torch.norm(all_prototype_fea, p=2, dim=1)).t()
    final_prototype_fea = all_prototype_fea.float().cpu().numpy()

    # split anchors by class
    anchor_fea_0 = final_prototype_fea[all_prototype_label == 0]  # (N0, D)
    anchor_fea_1 = final_prototype_fea[all_prototype_label == 1]  # (N1, D)

    start_test = True
    with torch.no_grad():
        for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(loader):
            for name in imgname:
                img_names.append(name)
            samples = images[0].to(device)
            for ii in range(len(labels)):
                labels[ii] = labels[ii].to(device)
            targets = labels[0].to(device)
            targetss = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
            if isinstance(masks[0], list):
                sample_masks = masks[0]
                for ii in range(len(sample_masks)):
                    sample_masks[ii] = sample_masks[ii].to(device)
            else:
                sample_masks = masks[0].to(device)
            images = (samples * sample_masks[0]).to(device)
            outputs, feas = model(images)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = targetss[:, 1].float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, targetss[:, 1].float().cpu()), 0)

    # -----------------------------
    # Softmax prediction
    # -----------------------------
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(
        predict.float() == all_label
    ).item() / float(all_label.size(0))

    # -----------------------------
    # Feature normalization
    # -----------------------------
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    final_fea = all_fea.float().cpu().numpy()

    # cosine distance: sample -> anchor set
    # shape: (N, N0), (N, N1)
    dd_anchor_0 = cdist(final_fea, anchor_fea_0, metric='cosine')
    dd_anchor_1 = cdist(final_fea, anchor_fea_1, metric='cosine')

    # take nearest anchor (more robust than mean)
    d0 = dd_anchor_0.min(axis=1)  # (N,)
    d1 = dd_anchor_1.min(axis=1)  # (N,)

    margin = np.zeros_like(d0)
    mask0 = (predict.cpu().numpy() == 0)
    mask1 = (predict.cpu().numpy() == 1)

    margin[mask0] = d1[mask0] - d0[mask0]
    margin[mask1] = d0[mask1] - d1[mask1]

    # threshold
    reliable_mask = np.zeros_like(margin, dtype=bool)

    # ---- class 0 ----
    if mask0.sum() > 0:
        m0 = margin[mask0]
        tau0 = np.maximum(0, np.percentile(m0, 20))  # 30 比 20 更稳
        reliable_mask[mask0] = m0 > tau0

    # ---- class 1 ----
    if mask1.sum() > 0:
        m1 = margin[mask1]
        tau1 = np.maximum(0, np.percentile(m1, 20))
        reliable_mask[mask1] = m1 > tau1

    final_output = all_output.float().cpu().numpy()
    final_label = all_label.float().cpu().numpy()
    predict = predict.cpu().numpy()
    reliable_mask_np = reliable_mask.astype(bool)
    fea_rel = final_fea[reliable_mask_np]
    out_rel = final_output[reliable_mask_np]
    pred_rel = predict[reliable_mask_np]
    label_rel = final_label[reliable_mask_np]

    K = final_output.shape[1]

    # -----------------------------
    # Define labelset (important!)
    # -----------------------------
    cls_count = np.eye(K)[pred_rel].sum(axis=0)
    labelset = np.where(cls_count > 0)[0]

    # =====================================================
    # Softmax-weighted + Per-class K-means Prototypes
    # =====================================================



    # ---------- Step 1: softmax-weighted single prototype ----------
    aff = out_rel
    initc_soft = aff.T @ fea_rel
    initc_soft = initc_soft / (1e-8 + aff.sum(axis=0)[:, None])
    # aff = final_output
    # initc_soft = aff.T @ final_fea
    # initc_soft = initc_soft / (1e-8 + aff.sum(axis=0)[:, None])

    # ---------- Step 2: per-class K-means prototypes ----------
    all_protos = []
    proto_labels = []

    for cls in labelset:
        idx = np.where(pred_rel == cls)[0]
        feats_cls = fea_rel[idx]

        if len(feats_cls) == 0:
            continue

        if len(feats_cls) <= NUM_PROTO:
            centers = feats_cls
        else:
            kmeans = KMeans(
                n_clusters=NUM_PROTO,
                random_state=0,
                n_init='auto'
            ).fit(feats_cls)
            centers = kmeans.cluster_centers_

        # 融合 softmax prototype（SHOT 原味）
        for c in centers:
            fused_center = INIT_WEIGHT * initc_soft[cls] + (1 - INIT_WEIGHT) * c
            # fused_center = INIT_WEIGHT * initc_prototype[cls] + (1 - INIT_WEIGHT) * fused_center0
            # fused_center = INIT_WEIGHT * initc_prototype[cls] + (1 - INIT_WEIGHT) * c
            all_protos.append(fused_center)
            proto_labels.append(cls)

    initc = np.vstack(all_protos)
    proto_labels = np.array(proto_labels)

    # ---------- Step 3: nearest prototype assignment ----------
    dd = cdist(final_fea, initc, 'cosine')
    nearest_proto = dd.argmin(axis=1)
    pred_label = proto_labels[nearest_proto]

    # ---------- Step 4: SHOT-style refinement (1 round) ----------
    for _ in range(1):
        all_protos = []
        proto_labels = []

        for cls in labelset:
            # idx = np.where(pred_label == cls)[0]
            # feats_cls = final_fea[idx]
            idx = np.where((pred_label == cls) & reliable_mask_np)[0]
            feats_cls = final_fea[idx]

            if len(feats_cls) == 0:
                continue

            if len(feats_cls) <= NUM_PROTO:
                centers = feats_cls
            else:
                kmeans = KMeans(
                    n_clusters=NUM_PROTO,
                    random_state=0,
                    n_init='auto'
                ).fit(feats_cls)
                centers = kmeans.cluster_centers_
            # all_protos.append(initc_prototype[cls])
            # proto_labels.append(cls)
            for c in centers:
                all_protos.append(c)
                proto_labels.append(cls)

        initc = np.vstack(all_protos)
        proto_labels = np.array(proto_labels)

        dd = cdist(final_fea, initc, 'cosine')
        nearest_proto = dd.argmin(axis=1)
        pred_label = proto_labels[nearest_proto]

    # -----------------------------
    # Final accuracy
    # -----------------------------
    acc = np.sum(pred_label == final_label) / len(final_fea)
    log_str = valdata + ' Accuracy = {:.2f}% -> {:.2f}%'.format(
        accuracy * 100, acc * 100
    )
    print(log_str + '\n')
    logger.info(log_str)
    model.train()
    return pred_label.astype(int), reliable_mask_np, img_names
from tqdm import tqdm
@torch.no_grad()
def validate_val_shot(data_loader, model, val_index='shanxi_val'):
    model.eval()
    start_test = True
    for idx, (images,masks, targets,_) in enumerate(data_loader):
        images = (images*masks).to(device)
        labels = targets.to(device)
        pred, feas = model(images)
        if start_test:
            all_fea = feas.float().cpu()
            all_output = pred.float().cpu()
            all_label = labels.float().cpu()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
            all_output = torch.cat((all_output, pred.float().cpu()), 0)
            all_label = torch.cat((all_label, labels.float().cpu()), 0)

    all_output = nn.Softmax(dim=-1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label[:,1]).item() / float(all_label.size()[0])
    predict_sick=predict[all_label[:, 1] == 0]
    all_label_sick=all_label[:,1][all_label[:, 1] == 0]
    predict_health = predict[all_label[:, 1] == 1]
    all_label_health = all_label[:, 1][all_label[:, 1] == 1]
    accuracy_sick = torch.sum(predict_sick.float() == all_label_sick).item() / float(all_label_sick.size()[0])
    accuracy_health = torch.sum(predict_health.float() == all_label_health).item() / float(all_label_health.size()[0])

    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label[:,1].float().numpy()) / len(all_fea)
    log_str = val_index+'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str)

    predict_sick = pred_label[all_label[:, 1] == 0]
    all_label_sick = all_label[:, 1][all_label[:, 1] == 0]
    predict_health = pred_label[all_label[:, 1] == 1]
    all_label_health = all_label[:, 1][all_label[:, 1] == 1]

    acc_sick = np.sum(predict_sick== all_label_sick.float().numpy()) / float(all_label_sick.size()[0])
    log_str = val_index + 'Sick Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy_sick * 100, acc_sick * 100)
    print(log_str)

    acc_health = np.sum(predict_health == all_label_health.float().numpy()) / float(all_label_health.size()[0])
    log_str = val_index + 'Health Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy_health * 100, acc_health * 100)
    print(log_str + '\n')
    return pred_label.astype('int') #, labelset
if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)

    # model_name = ['pneullm',]
    # model_name = ['slgms', ]
    # model_name = ['dinov3']
    # model_name = ['qwen2_5vision']
    model_name = ['convnext']
    lrs=[8e-5]
    radios = [15]
    num_prototypes=[4]
    LOCRADIOS=[0.6]

    for ii in range(len(radios)):
        for jj in range(len(num_prototypes)):
            for kk in range(len(LOCRADIOS)):
                args = parse_option()
                args.model_name = model_name[0]
                config = get_config(args)
                config.TRAIN.EPOCHS = 5
                config.TRAIN.WARMUP_EPOCHS = 0
                config.DATA.BATCH_SIZE = 10
                config.DATA.BATCH_SIZE_SMALL = 10
                config.BASE_LR = lrs[0]
                radio = radios[ii]
                config.RADIO=radio
                num_prototype=num_prototypes[jj]
                config.NUM_PROTOTYPES=num_prototype
                config.LOCRADIO=LOCRADIOS[kk]
                config.OUTPUT='/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CALGM/ablation/para_trainable/first15_trainable'

                # pt_path='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/trainable_bestvalpt.pth'
                # pt_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
                pt_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
                # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-b/batch16_lr1e-05_testall/model_best_acc_shanxi_test.pth'
                # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/2models_distill/dinov3stu224_trained_convnext_slgmstea/224_batch10_lr1e-05_testall/model_best_acc_alltest.pth'
                # pt_path = '/di/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/sk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout/3models_distill/dinov3st224_notrained_convnext_slgms_pneullmtea_0.5weight/224_batch10_lr3e-05_testall/model_best_acc_alltest.pth'
                # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_3B/best_224_batch16_lr1e-05_testall/model_best_acc_shanxi_val.pth'

                folder_path = config.OUTPUT
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name = 'shot_biaozhun_reliable_multi_prototypes_glb_locresult.txt'
                file_path = os.path.join(folder_path, file_name)
                logger_path = os.path.join(config.OUTPUT,
                                         f'15batch_{config.BASE_LR}lr_{config.RADIO}lpara_{config.NUM_PROTOTYPES}prototypes_{config.LOCRADIO}rule_JSD')
                if not os.path.exists(logger_path):
                    os.makedirs(logger_path)
                logger = create_logger(output_dir=logger_path, name=f"{config.MODEL.NAME}")

                path = os.path.join(logger_path, "config.json")
                with open(path, "w") as f:
                    f.write(config.dump())
                logger.info(f"Full config saved to {path}")
                shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot = main(config, pt_path,logger_path)
                avg_test_acc_glb_org = (shanxi_test_org[0][1] * 3847 + guizhou_test_org[0][1] * 3144) / 6991
                avg_test_acc_glb_tent = (shanxi_test_shot[0][1] * 3847 + guizhoutest_shot[0][1] * 3144) / 6991
                avg_test_acc_loc_org = (shanxi_test_org[1][1] * 3847 + guizhou_test_org[1][1] * 3144) / 6991
                avg_test_acc_loc_tent = (shanxi_test_shot[1][1] * 3847 + guizhoutest_shot[1][1] * 3144) / 6991
                avg_test_acc_glb_loc_org = (shanxi_test_org[2][1] * 3847 + guizhou_test_org[2][1] * 3144) / 6991
                avg_test_acc_glb_loc_tent = (shanxi_test_shot[2][1] * 3847 + guizhoutest_shot[2][1] * 3144) / 6991

                with open(file_path, 'a', encoding='utf-8') as file:
                    re = f"{args.model_name} {config.BASE_LR} learnable_para_RADIO:{config.RADIO} NUM_PROTOTYPES:{config.NUM_PROTOTYPES}: JSDnew_sum_rule_weight:{config.LOCRADIO}\n"
                        # --- Shanxi ORG ---
                    re += "=== Shanxi Test (ORG) ===\n"
                    # global
                    g = shanxi_test_org[0]
                    re += f"GLOBAL      → avg: {g[0]:.4f}, acc: {g[1]:.4f}, sen: {g[2]:.4f}, spe: {g[3]:.4f}, auc: {g[4]:.4f}\n"
                    # local
                    l = shanxi_test_org[1]
                    re += f"LOCAL       → avg: {l[0]:.4f}, acc: {l[1]:.4f}, sen: {l[2]:.4f}, spe: {l[3]:.4f}, auc: {l[4]:.4f}\n"
                    # local_global
                    lg = shanxi_test_org[2]
                    re += f"LOCAL_GLOBAL→ avg: {lg[0]:.4f}, acc: {lg[1]:.4f}, sen: {lg[2]:.4f}, spe: {lg[3]:.4f}, auc: {lg[4]:.4f}\n\n"

                    # --- Guizhou ORG ---
                    re += "=== Guizhou Test (ORG) ===\n"
                    g = guizhou_test_org[0]
                    re += f"GLOBAL      → avg: {g[0]:.4f}, acc: {g[1]:.4f}, sen: {g[2]:.4f}, spe: {g[3]:.4f}, auc: {g[4]:.4f}\n"
                    l = guizhou_test_org[1]
                    re += f"LOCAL       → avg: {l[0]:.4f}, acc: {l[1]:.4f}, sen: {l[2]:.4f}, spe: {l[3]:.4f}, auc: {l[4]:.4f}\n"
                    lg = guizhou_test_org[2]
                    re += f"LOCAL_GLOBAL→ avg: {lg[0]:.4f}, acc: {lg[1]:.4f}, sen: {lg[2]:.4f}, spe: {lg[3]:.4f}, auc: {lg[4]:.4f}\n\n"

                    # --- Shanxi SHOT ---
                    re += "=== Shanxi Test (SHOT) ===\n"
                    g = shanxi_test_shot[0]
                    re += f"GLOBAL      → avg: {g[0]:.4f}, acc: {g[1]:.4f}, sen: {g[2]:.4f}, spe: {g[3]:.4f}, auc: {g[4]:.4f}\n"
                    l = shanxi_test_shot[1]
                    re += f"LOCAL       → avg: {l[0]:.4f}, acc: {l[1]:.4f}, sen: {l[2]:.4f}, spe: {l[3]:.4f}, auc: {l[4]:.4f}\n"
                    lg = shanxi_test_shot[2]
                    re += f"LOCAL_GLOBAL→ avg: {lg[0]:.4f}, acc: {lg[1]:.4f}, sen: {lg[2]:.4f}, spe: {lg[3]:.4f}, auc: {lg[4]:.4f}\n\n"

                    # --- Guizhou SHOT ---
                    re += "=== Guizhou Test (SHOT) ===\n"
                    g = guizhoutest_shot[0]
                    re += f"GLOBAL      → avg: {g[0]:.4f}, acc: {g[1]:.4f}, sen: {g[2]:.4f}, spe: {g[3]:.4f}, auc: {g[4]:.4f}\n"
                    l = guizhoutest_shot[1]
                    re += f"LOCAL       → avg: {l[0]:.4f}, acc: {l[1]:.4f}, sen: {l[2]:.4f}, spe: {l[3]:.4f}, auc: {l[4]:.4f}\n"
                    lg = guizhoutest_shot[2]
                    re += f"LOCAL_GLOBAL→ avg: {lg[0]:.4f}, acc: {lg[1]:.4f}, sen: {lg[2]:.4f}, spe: {lg[3]:.4f}, auc: {lg[4]:.4f}\n\n"

                    # --- Overall Weighted Accuracy (by sample count: Shanxi=3847, Guizhou=3144) ---
                    re += "=== OVERALL WEIGHTED ACCURACY (Shanxi + Guizhou) ===\n"
                    re += f"ORG:\n"
                    re += f"  GLOBAL       → {avg_test_acc_glb_org:.4f}\n"
                    re += f"  LOCAL        → {avg_test_acc_loc_org:.4f}\n"
                    re += f"  LOCAL_GLOBAL → {avg_test_acc_glb_loc_org:.4f}\n"
                    re += f"SHOT:\n"
                    re += f"  GLOBAL       → {avg_test_acc_glb_tent:.4f}\n"
                    re += f"  LOCAL        → {avg_test_acc_loc_tent:.4f}\n"
                    re += f"  LOCAL_GLOBAL → {avg_test_acc_glb_loc_tent:.4f}\n"
                    re += "=" * 80 + "\n\n"

                    file.write(re)





