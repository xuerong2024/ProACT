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
sys.path.append('/disk3/wjr/workspace/sec_proj4/proj4_baseline/')
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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda:4')
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
    val_transform1 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    global_transfo = transforms.Compose([
        transforms.Resize((1024, 1024)),
    ])
    global_transfo_subregions = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil2tensor_transfo = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
    dataset_prototype = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                              args.data_root + 'seg_rec_mask_1024',
                                                              txtpath=args.data_root + '/chinese_biaozhun.txt',
                                                              csvpath=args.data_root + 'subregions_label_shanxi_all_woyinying.xlsx',
                                                              data_transform=global_transfo,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo_subregions,
                                                              sub_img_size=256, )
    dataset_test = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                              args.data_root + 'seg_rec_mask_1024',
                                                              txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                                              csvpath=args.data_root + 'subregions_label_shanxi_all_woyinying.xlsx',
                                                              data_transform=global_transfo,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo_subregions,
                                                              sub_img_size=256, )
    dataset_test2 = Shanxi_w7masks_5Subregions_wsubroi_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024','/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                                              txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                                              csvpath='/disk3/wjr/dataset/nejm/guizhoudataset/subregions_guizhou_all.xlsx',
                                                              data_transform=global_transfo,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo_subregions,
                                                              sub_img_size=256, )
    dataset_test_glb = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + '/seg_rec_mask_1024',
                                            txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                            data_transform=val_transform1)
    dataset_test2_glb = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                             '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                             txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                             data_transform=val_transform1)

    data_loader_prototype = torch.utils.data.DataLoader(
        dataset_prototype,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.SMALL_BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2,
        batch_size=config.DATA.SMALL_BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_glb_loader_test = torch.utils.data.DataLoader(
        dataset_test_glb,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_glb_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2_glb,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    return data_loader_prototype, data_loader_test, data_loader_test2, data_glb_loader_test, data_glb_loader_test2

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
def main(config, ptname):
    data_loader_val, data_loader_test, data_loader_test2, data_glb_loader_test, data_glb_loader_test2 = build_phe_loader(args)
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

        model = train_one_epoch_glb(config, model, data_loader_test, data_glb_loader_test)
        shanxi_test_shot=validate_network_glb(data_glb_loader_test, model, val_index='shanxi_test')
        if args.model_name == "slgms":
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['teacher']
            state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
            msg = model.load_state_dict(state_dict, strict=False)
        shanxi_test_org = validate_network_glb(data_glb_loader_test, model, 'shanxi_test')
        guizhou_test_org = validate_network_glb(data_glb_loader_test2, model, 'guizhou_test')

        model = train_one_epoch_glb(config, model, data_loader_test2, data_glb_loader_test2)
        guizhoutest_shot=validate_network_glb(data_glb_loader_test2, model, val_index='guizhou_test')
        # obtain_label(data_loader_val, model,'shanxival')
        # obtain_label(data_loader_test, model, 'shanxitest')
        # obtain_label(data_loader_test2, model, 'guizhoutest')
        return shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot


import operator


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
@torch.no_grad()
def validate_network_glb(val_loader, model, val_index='shanxi_val'):
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
@torch.no_grad()
def validate_network(val_loader, model, val_index='shanxi_val', tau=0.2, alpha=0.5):
    model.eval()
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    data_results = []
    total_loss=0
    id=0
    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(val_loader):
        result = {}
        result["Image Index"] = imgname[0]
        result["Finding Labels"] = imgname[0].split('_')[0]
        id += 1
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
        outputs, feas = model(images)
        glb_prob = torch.nn.functional.softmax(outputs, dim=1)
        subregions_outputs, subfeas = model(subregions_imgs * subregions_masks)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
        subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
        # 规则推理
        rule_prob = local2global_prob(subregions_outputs2)
        diff = torch.abs(glb_prob[:, 1] - rule_prob[:, 1])
        w = (diff < tau).float().unsqueeze(1)
        final_prob = alpha * glb_prob + (1 - alpha) * rule_prob
        final_prob = w * final_prob + (1 - w) * glb_prob

        for bb in range(final_prob.shape[0]):
            aa = torch.argmax(final_prob[bb, :])
            if aa == 0:
                pred_global_labels = 'Sick'
            else:
                pred_global_labels = 'Health'
            result["Pred Labels"] = pred_global_labels
            result["Pred logits"] = '%.4f' % final_prob[bb, aa].cpu().detach().numpy()
            data_results.append(result)
            # 收集输出和目标
            all_outputs.extend(final_prob[bb, :].unsqueeze(0).data.cpu().numpy())
            # targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

            all_targets.extend(targetss[bb, :].unsqueeze(0).cpu().numpy())

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
def train_one_epoch(config, model, data_loader, tau=0.2):
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    optimizer = build_optimizer(config, model)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader))

    for k, v in model.named_parameters():
        # v.requires_grad = True
        if k.__contains__("downsample_layers"):
            v.requires_grad = True
        else:
            v.requires_grad = False

    logger.info("Start training")

    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    for epoch in range(config.TRAIN.EPOCHS):
        model.eval()
        # mem_label, imgnames = obtain_label(data_loader, model)
        mem_label_certain, mem_label_uncertain, mem_label, imgnames = obtain_sub_label_uncertainty(data_loader, model)
        mem_label = torch.from_numpy(mem_label).to(device)
        mem_label_certain = torch.from_numpy(mem_label_certain).to(device)
        mem_label_uncertain = torch.from_numpy(mem_label_uncertain).to(device)
        # mem_sub_label = torch.from_numpy(mem_sub_label).to(device)
        model.train()
        loss_meter = AverageMeter()
        loss_local_meter = AverageMeter()
        loss_entropy_meter = AverageMeter()
        loss_consistency_meter = AverageMeter()
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
            outputs, feas = model(images)
            glb_prob=torch.nn.functional.softmax(outputs, dim=1)
            subregions_outputs, subfeas = model(subregions_imgs * subregions_masks)
            subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
            subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
            # 规则推理
            rule_prob = local2global_prob(subregions_outputs2)
            conf_mask = torch.abs(glb_prob[:, 1] - rule_prob[:, 1]) < tau
            if conf_mask.sum() > 0:
                L_ent_med = Entropy(glb_prob[conf_mask]).mean()
            else:
                L_ent_med = torch.tensor(0.0, device=device)
            pred = mem_label[idx * config.DATA.BATCH_SIZE:idx * config.DATA.BATCH_SIZE + images.shape[0]]
            L_classifier_loss = nn.CrossEntropyLoss()(outputs, pred)
            L_classifier_local_loss = F.nll_loss(torch.log(rule_prob + 1e-8), pred)
            kl_per_sample=F.kl_div(torch.log(glb_prob[conf_mask,:] + 1e-8),rule_prob[conf_mask,:].detach(),reduction='none')
            # 2. 创建掩码：只保留 loss >= 0.02 的样本
            mask = kl_per_sample >= 0.02  # [B]
            # 3. 只对这些样本求平均（避免 nan）
            if mask.any():
                L_consistency = kl_per_sample[mask].mean()
            else:
                L_consistency = torch.tensor(0.0, device=glb_prob.device)
            loss = L_classifier_loss+0.1*L_consistency + L_ent_med * 0.01
            # loss = classifier_loss * 0.3 + entropy_loss * 1.
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss.backward()
            optimizer.step()
            loss_meter.update(L_classifier_loss.item(), images.size(0))
            loss_local_meter.update(L_classifier_local_loss.item(), images.size(0))
            loss_entropy_meter.update(L_ent_med.item(), images.size(0))
            loss_consistency_meter.update(L_consistency.item(), images.size(0))
            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'classifier_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'local_classifier_loss {loss_local_meter.val:.4f} ({loss_local_meter.avg:.4f})\t'
                    f'entropy_loss {loss_entropy_meter.val:.4f} ({loss_entropy_meter.avg:.4f})\t'
                    f'consistency_loss {loss_consistency_meter.val:.4f} ({loss_consistency_meter.avg:.4f})\t'
                )
    mem_label, imgnames = obtain_sub_label(data_loader, model)
    return model


def train_one_epoch_glb(config, model, data_loader,data_loader_glb, tau=0.1):
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    optimizer = build_optimizer(config, model)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader))

    for k, v in model.named_parameters():
        # v.requires_grad = True
        if k.__contains__("downsample_layers"):
            v.requires_grad = True
        else:
            v.requires_grad = False

    logger.info("Start training")

    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    for epoch in range(config.TRAIN.EPOCHS):
        model.eval()
        # mem_label, imgnames = obtain_label(data_loader, model)
        mem_label_certain, mem_label_uncertain, mem_label, imgnames = obtain_sub_label_uncertainty(data_loader, model)
        mem_label = torch.from_numpy(mem_label).to(device)
        mem_label_certain = torch.from_numpy(mem_label_certain).to(device)
        mem_label_uncertain = torch.from_numpy(mem_label_uncertain).to(device)
        # mem_sub_label = torch.from_numpy(mem_sub_label).to(device)
        model.train()
        loss_certain_meter = AverageMeter()
        loss_uncertain_meter = AverageMeter()
        for idx, (images, masks, targetss, img_names) in enumerate(data_loader_glb):
            images = (images * masks).to(device)
            targetss = targetss.to(device)
            outputs, feas = model(images)
            # glb_prob=torch.nn.functional.softmax(outputs, dim=1)
            mem_labelii=mem_label[idx * config.DATA.BATCH_SIZE:idx * config.DATA.BATCH_SIZE + images.shape[0]]
            pred = mem_label_certain[idx * config.DATA.BATCH_SIZE:idx * config.DATA.BATCH_SIZE + images.shape[0]]
            mask=pred!=-1
            if mask.sum() > 0:
                L_classifier_loss_certain = nn.CrossEntropyLoss()(outputs[mask,:], mem_labelii[mask])
            else:
                L_classifier_loss_certain = torch.tensor(0.0, device=device)
            pred_uncertain = mem_label_uncertain[idx * config.DATA.BATCH_SIZE:idx * config.DATA.BATCH_SIZE + images.shape[0]]
            mask_uncertain = pred_uncertain != -1
            if mask_uncertain.sum()>0:
                L_classifier_loss_uncertain = nn.CrossEntropyLoss()(outputs[mask_uncertain, :],
                                                                    mem_labelii[mask_uncertain])
            else:
                L_classifier_loss_uncertain = torch.tensor(0.0, device=device)
            loss = L_classifier_loss_certain+0.3*L_classifier_loss_uncertain
            # loss = classifier_loss * 0.3 + entropy_loss * 1.
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss.backward()
            optimizer.step()
            loss_certain_meter.update(L_classifier_loss_certain.item(), images.size(0))
            loss_uncertain_meter.update(L_classifier_loss_uncertain.item(), images.size(0))

            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'classifier_certain_loss {loss_certain_meter.val:.4f} ({loss_certain_meter.avg:.4f})\t'
                    f'classifier_uncertain_loss {loss_uncertain_meter.val:.4f} ({loss_uncertain_meter.avg:.4f})\t'
                )
    mem_label, imgnames = obtain_sub_label(data_loader, model)
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
def compute_reliability(glb_prob, rule_prob, tau=0.1, conf_threshold=0.8):
    """
    返回：
      reliable_mask: Bool [B]
      conf: max softmax
      diff: |p_glb - p_rule|
    """
    conf, _ = glb_prob.max(dim=1)
    diff = torch.abs(glb_prob[:, 1] - rule_prob[:, 1])
    reliable_mask = (conf > conf_threshold) & (diff < tau)
    return reliable_mask, conf, diff

@torch.no_grad()
def obtain_sub_label(loader, model, valdata='', tau=0.2, alpha=0.5):
    model.eval()
    start_test = True
    img_names=[]
    with torch.no_grad():
        for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(loader):
            for iimgname in imgname:
                img_names.append(iimgname)
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
            outputs, feas = model(images)
            glb_prob=torch.nn.functional.softmax(outputs, dim=1)
            subregions_outputs, subfeas = model(subregions_imgs * subregions_masks)
            subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
            subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
            # 规则推理
            rule_prob = local2global_prob(subregions_outputs2)
            # ========= 2. 医学一致性加权概率 =========
            # 一致样本：更相信规则
            diff = torch.abs(glb_prob[:, 1] - rule_prob[:, 1])
            w = (diff < tau).float().unsqueeze(1)
            final_prob = alpha * glb_prob + (1 - alpha) * rule_prob
            final_prob = w * final_prob + (1 - w) * glb_prob
            if start_test:
                all_fea = [{f'{idx}': feas.float().cpu()}]
                all_output = [{f'{idx}': final_prob.float().cpu()}]
                all_label = [{f'{idx}': targetss[:,1].float().cpu()}]
                start_test = False
            else:
                all_fea.append({f'{idx}': feas.float().cpu()})
                all_output.append({f'{idx}': final_prob.float().cpu()})
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
@torch.no_grad()
def obtain_sub_label_uncertainty(loader, model, valdata='', conf_threshold=0.7, tau=0.1, alpha=0.5):
    model.eval()
    start_test = True
    img_names=[]
    reliable_flags=[]
    with torch.no_grad():
        for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(loader):
            # if idx>5:
            #     break
            for iimgname in imgname:
                img_names.append(iimgname)
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
            outputs, feas = model(images)
            glb_prob=torch.nn.functional.softmax(outputs, dim=1)
            subregions_outputs, subfeas = model(subregions_imgs * subregions_masks)
            subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
            subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
            # 规则推理
            rule_prob = local2global_prob(subregions_outputs2)
            # ========= 2. 医学一致性加权概率 =========
            reliable_mask, _, _ = compute_reliability(glb_prob, rule_prob, tau, conf_threshold)
            reliable_flags.append(reliable_mask.cpu())
            w = reliable_mask.unsqueeze(-1).float()
            final_prob = alpha * glb_prob + (1 - alpha) * rule_prob
            final_prob = w * final_prob + (1 - w) * glb_prob
            if start_test:
                all_fea = [{f'{idx}': feas.float().cpu()}]
                all_output = [{f'{idx}': final_prob.float().cpu()}]
                all_label = [{f'{idx}': targetss[:,1].float().cpu()}]
                start_test = False
            else:
                all_fea.append({f'{idx}': feas.float().cpu()})
                all_output.append({f'{idx}': final_prob.float().cpu()})
                all_label.append({f'{idx}': targetss[:,1].float().cpu()})

    # 分别合并 fea、output、label
    final_fea = merge_dict_list(all_fea)
    final_output = merge_dict_list(all_output)
    final_output = nn.Softmax(dim=1)(final_output)
    final_label = merge_dict_list(all_label)
    reliable_flags = torch.cat(reliable_flags).numpy()
    final_fea = final_fea.float().cpu().numpy()

    _, predict = torch.max(final_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == final_label).item() / float(final_label.size()[0])

    K = final_output.size(1)
    # ===== Step 1: 只用可靠样本初始化 prototype =====
    initc = np.zeros((K, final_fea.shape[1]))
    pred_class = final_output.argmax(axis=1)  # 先算一次，避免重复计算
    for k in range(K):
        mask = (pred_class == k) & reliable_flags
        if mask.sum() > 0:
            initc[k] = final_fea[mask==1,:].mean(0)

    # ===== Step 2: masked K-means =====
    pred_label = -np.ones(len(final_fea), dtype=int)
    pred_label_uncertain = -np.ones(len(final_fea), dtype=int)

    for _ in range(1):
        dist = cdist(final_fea, initc, metric='cosine')
        new_label = dist.argmin(1)

        # 只更新可靠样本
        for k in range(K):
            mask = (new_label == k) & reliable_flags
            if mask.sum() > 0:
                initc[k] = final_fea[mask==1,].mean(0)

        pred_label[mask] = new_label[mask]
        pred_label_uncertain[~mask] = new_label[~mask]

    acc = np.sum(new_label == final_label.float().numpy()) / len(final_fea)
    log_str = valdata+' Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')
    model.train()
    return pred_label,pred_label_uncertain, new_label, img_names #, labelset
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
    lrs=[1e-5]
    for kk in range(len(lrs)):
        args = parse_option()
        args.model_name = model_name[0]
        config = get_config(args)
        config.TRAIN.EPOCHS = 5
        config.TRAIN.WARMUP_EPOCHS = 0
        config.DATA.BATCH_SIZE = 30
        config.DATA.SMALL_BATCH_SIZE = 10
        config.BASE_LR = lrs[kk]

        # pt_path='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/trainable_bestvalpt.pth'
        # pt_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
        pt_path ='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-b/batch16_lr1e-05_testall/model_best_acc_shanxi_test.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/2models_distill/dinov3stu224_trained_convnext_slgmstea/224_batch10_lr1e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout/3models_distill/dinov3st224_notrained_convnext_slgms_pneullmtea_0.5weight/224_batch10_lr3e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_3B/best_224_batch16_lr1e-05_testall/model_best_acc_shanxi_val.pth'
        config.OUTPUT = pt_path.split(pt_path.split('/')[-1])[0]
        folder_path = config.OUTPUT
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = 'shot_result.txt'
        file_path = os.path.join(folder_path, file_name)
        logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot=main(config, pt_path)


        with open(file_path, 'a', encoding='utf-8') as file:
            re = f'{args.model_name} {config.BASE_LR}:\n' \
                 f'ORG global_shanxi_test_avg: {shanxi_test_org[0]}, shanxi_test_auc: {shanxi_test_org[-1]}, shanxi_test_acc: {shanxi_test_org[1]}, shanxi_test_sensitivity: {shanxi_test_org[2]}, shanxi_test_specificity: {shanxi_test_org[3]}\n' \
                 f'ORG global_guizhou_test_avg: {guizhou_test_org[0]}, guizhou_test_auc: {guizhou_test_org[-1]}, guizhou_test_acc: {guizhou_test_org[1]}, guizhou_test_sensitivity: {guizhou_test_org[2]}, guizhou_test_specificity: {guizhou_test_org[3]}\n' \
                 f'SHOT TRAIN_FEA_local2global global_shanxi_test_avg: {shanxi_test_shot[0]}, shanxi_test_auc: {shanxi_test_shot[-1]}, shanxi_test_acc: {shanxi_test_shot[1]}, shanxi_test_sensitivity: {shanxi_test_shot[2]}, shanxi_test_specificity: {shanxi_test_shot[3]}\n' \
                 f'SHOT TRAIN_FEA_local2global global_guizhou_test_avg: {guizhoutest_shot[0]}, guizhou_test_auc: {guizhoutest_shot[-1]}, guizhou_test_acc: {guizhoutest_shot[1]}, guizhou_test_sensitivity: {guizhoutest_shot[2]}, guizhou_test_specificity: {guizhoutest_shot[3]}\n\n'

            file.write(re)

