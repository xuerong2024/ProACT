# Contrastive Test-Time Adaptation, CVPR2022, 随机训练一小部分模型参数
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
from test_adaptation.cotta_transform import *
from copy import deepcopy
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor
def build_phe_loader(args):
    # 设置随机数种子
    setup_seed(args.seed)
    val_transform1 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + '/seg_rec_mask_1024',
                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                        data_transform=val_transform1)
    # dataset_test = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024',args.data_root + '/seg_rec_mask_1024',
    #                               txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
    #                               data_transform=val_transform1)
    # dataset_test2 = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024','/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
    #                                txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
    #                                data_transform=val_transform1)
    dataset_test = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + '/seg_rec_mask_1024',
                                        txtpath='/disk3/wjr/dataset/nejm/shanxi_test_chosen/random100_2.txt',
                                        data_transform=val_transform1)
    dataset_test2 = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                         '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                         txtpath='/disk3/wjr/dataset/nejm/guizhou_test_chossen/guizhou_100_2.txt',
                                         data_transform=val_transform1)
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

    return data_loader_val, data_loader_test, data_loader_test2

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
    start_time = time.time()
    data_loader_val, data_loader_test, data_loader_test2 = build_phe_loader(args)
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

        model, bank = train_one_epoch(config, model, data_loader_test, val_index='shanxi_test')
        shanxi_test_shot=validate_network(data_loader_test, model, bank, val_index='shanxi_test', save_path=logger_path+'/analyze/')
        end_time = time.time()
        # 总耗时（包含数据加载）
        total_inference_time = end_time - start_time
        print(f"{args.model_name}: Total inference time (including data loading): {total_inference_time:.4f} seconds")
        logger.info(
            f"{args.model_name}: Total inference time (including data loading): {total_inference_time:.4f} seconds")

        save_state = {'model': model.state_dict(), 'bank': bank}
        save_path = os.path.join(logger_path, 'shanxi_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        gc.collect()

        if args.model_name == "slgms":
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['teacher']
            state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
            model.head.fc = nn.Linear(768, 2)
            state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
            msg = model.load_state_dict(state_dict, strict=False)
        model.to(device)

        shanxi_test_org = validate_network_org(data_loader_test, model, 'shanxi_test_org', save_path=logger_path+'/analyze/')
        guizhou_test_org = validate_network_org(data_loader_test2, model, 'guizhou_test_org', save_path=logger_path+'/analyze/')
        model, bank = train_one_epoch(config, model, data_loader_test2, val_index='guizhou_test')
        guizhoutest_shot=validate_network(data_loader_test2, model, bank, val_index='guizhou_test', save_path=logger_path+'/analyze/')
        save_state = {'model': model.state_dict(), 'bank': bank}
        save_path = os.path.join(logger_path, 'guizhou_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        # obtain_label(data_loader_val, model,'shanxival')
        # obtain_label(data_loader_test, model, 'shanxitest')
        # obtain_label(data_loader_test2, model, 'guizhoutest')
        return shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot, total_inference_time


import operator
def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def update_labels(banks, features, logits):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + features.shape[0]
    idxs_replace = torch.arange(start, end).to(device) % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])
    return banks



@torch.no_grad()
def refine_predictions(
    features,
    banks,
    gt_labels=None,
):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank
    )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100
        print(f"Accuracy of refine prediction: {accuracy:.2f}")

    return pred_labels, probs, accuracy

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors=10):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, 'cosine')
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


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

def adaptive_calculate_metrics(all_outputs, all_targets_onehot,threshold=None, valdata='shanxi_test',num_classes=2, plot_roc=True, save_fpr_tpr=True, save_path=''):
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
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc * 100:.2f}' + '%')
            plt.plot([0, 1], [0, 1], 'k--')
            # 找到最接近特定 threshold 的索引
            idx = np.argmin(np.abs(thresholds - threshold))
            # 获取对应的 fpr 和 tpr 值
            specific_fpr = fpr[idx]
            specific_tpr = tpr[idx]
            # 使用 plt.scatter 标记特定点
            plt.scatter(specific_fpr, specific_tpr, marker='*', color='red', s=200, zorder=10)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            plt.legend(loc='lower right', prop={'size': 20})
            imsave_path = save_path+ '/' + valdata + '_roc_map.pdf'
            plt.savefig(imsave_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
            # # # 显示
            # # plt.show()
            # # 自动关闭图像
            plt.close()

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
        np.save(save_path + '/' + valdata + '_fpr.npy', fpr)
        np.save(save_path + '/' + valdata + '_tpr.npy', tpr)
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
def validate_network(val_loader, model, banks, val_index='shanxi_val', save_path=''):
    model.eval()
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_outputs_org = []
    all_targets = []
    data_results = []
    total_loss=0
    id=0
    for i, (inp, mask, target, imgname) in enumerate(val_loader):

        id+=1
        # move to gpu
        inp = (inp* mask).to(device)
        # inp = inp.to(device)
        # inp = inp.cuda(non_blocking=True)
        target = target.to(device)

        # forward
        with torch.no_grad():
            output, fea = model(inp, cls_only=True)
            # fea=F.normalize(fea, dim=1)
            # refine predicted labels
            pred_labels, prob, _ = refine_predictions(fea, banks)
        prob_org = torch.nn.functional.softmax(output, dim=1).float()

        for bb in range(prob.shape[0]):
            result = {}
            result["Image Index"] = imgname[bb]
            result["Finding Labels"] = imgname[bb].split('_')[0]
            aa = torch.argmax(prob[bb,:])
            if aa == 0:
                pred_global_labels = 'Sick'
            else:
                pred_global_labels = 'Health'
            result["Pred Labels"] = pred_global_labels
            result["Pred logits"] = '%.4f' % prob[bb, aa].cpu().detach().numpy()
            data_results.append(result)
            # 收集输出和目标
            all_outputs.extend(prob[bb,:].unsqueeze(0).data.cpu().numpy())
            all_outputs_org.extend(prob_org[bb, :].unsqueeze(0).data.cpu().numpy())
            # targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

            all_targets.extend(target[bb,:].unsqueeze(0).cpu().numpy())

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_outputs_org = np.array(all_outputs_org)
    all_targets = np.array(all_targets)
    df_data_results = pd.DataFrame(data_results)
    df_data_results.to_csv(save_path + '/' + val_index + '_dataprob_result.csv', index=False)

    # df_data_results = pd.DataFrame(data_results)
    # Path(config.OUTPUT + '/' + 'analyze').mkdir(parents=True, exist_ok=True)
    # df_data_results.to_csv(config.OUTPUT + '/' + val_index + '_dataprob_result.csv', index=False)
    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs_org,
        all_targets,
        threshold=0.5,
        valdata=val_index+'_worefine',
        plot_roc=True,
        save_fpr_tpr=True,
        save_path=save_path
    )

    avg = (acc + auc + specificity + sensitivity) / 4
    print(
        f'{val_index} ORG Sensitivity: {sensitivity}  Specificity: {specificity}, Accuracy: {acc}  AUC: {auc}, avg: {avg} ')

    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=val_index,
        plot_roc=True,
        save_fpr_tpr=True,
    save_path=save_path)
    avg = (acc + auc + specificity + sensitivity) / 4
    print(f'{val_index} Sensitivity: {sensitivity}  Specificity: {specificity}, Accuracy: {acc}  AUC: {auc}, avg: {avg} ')
    logger.info(f'{val_index} Sensitivity: {sensitivity}  Specificity: {specificity}, Accuracy: {acc}  AUC: {auc}, avg: {avg} ')

    # logger.info(
    #     f'lr: {args.base_lr} {valdata} {ptnames}' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)
    return avg, acc, sensitivity, specificity, auc
    # return loss_meter.avg, auc, acc, writer
@torch.no_grad()
def validate_network_org(val_loader, model, val_index='shanxi_val', save_path=''):
    model.eval()
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    data_results = []
    total_loss=0
    id=0
    for i, (inp, mask, target, imgname) in enumerate(val_loader):

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
                output = output[0]
            prob = torch.nn.functional.softmax(output, dim=1).float()

        for bb in range(prob.shape[0]):
            result = {}
            result["Image Index"] = imgname[bb]
            result["Finding Labels"] = imgname[bb].split('_')[0]
            aa = torch.argmax(prob[bb,:])
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
    df_data_results = pd.DataFrame(data_results)
    df_data_results.to_csv(save_path + '/' + val_index + '_dataprob_result.csv', index=False)
    # df_data_results = pd.DataFrame(data_results)
    # Path(config.OUTPUT + '/' + 'analyze').mkdir(parents=True, exist_ok=True)
    # df_data_results.to_csv(config.OUTPUT + '/' + val_index + '_dataprob_result.csv', index=False)

    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=val_index,
        plot_roc=True,
        save_fpr_tpr=True, save_path=save_path)
    avg = (acc + auc + specificity + sensitivity) / 4
    print(f'{val_index} Sensitivity: {sensitivity}  Specificity: {specificity}, Accuracy: {acc}  AUC: {auc}, avg: {avg} ')
    # logger.info(
    #     f'lr: {args.base_lr} {valdata} {ptnames}' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)
    return avg, acc, sensitivity, specificity, auc
    # return loss_meter.avg, auc, acc, writer
@torch.no_grad()
def eval_and_label_dataset(dataloader, model, queue_size):
    # wandb_dict = dict()

    # make sure to switch to eval mode
    model.eval()

    # run inference
    logits, gt_labels, indices = [], [], []
    features = []
    logging.info("Eval and labeling...")
    for idxs, (images, masks, targetss, img_names) in enumerate(dataloader):
        # move to gpu
        inp = (images * masks).to(device)
        # inp = inp.to(device)
        # inp = inp.cuda(non_blocking=True)
        labels = targetss.to(device)

        # (B, D) x (D, K) -> (B, K)
        logits_cls, feats = model(inp, cls_only=True)
        # feats = F.normalize(feats, dim=1)
        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels[:,1])
        indices.append(idxs)

    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to(device)

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    print(f"Accuracy of direct prediction: {accuracy:.2f}")

    probs = F.softmax(logits, dim=1)
    banks = {
        "features": features[: queue_size],
        "probs": probs[: queue_size],
        "ptr": 0,
    }
    # banks = {
    #     "features": features[rand_idxs][: queue_size],
    #     "probs": probs[rand_idxs][: queue_size],
    #     "ptr": 0,
    # }

    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
        features, banks, gt_labels=gt_labels
    )
    return banks

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div
def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type= "class_aware"):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(device)

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(device))

    loss = F.cross_entropy(logits_ins, labels_ins)
    return loss
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return tensor  # 单卡无需 gather
class AdaMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a memory bank
    https://arxiv.org/abs/1911.05722
    """
    def __init__(
        self,
        src_model,
        momentum_model,
        K=16384,
        feature_dim=768,
        num_classes=2,
        m=0.999,
        T_moco=0.07
    ):
        """
        dim: feature dimension (default: 128)
        K: buffer size; number of keys
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdaMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K))
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (K,))
        )
        self.mem_feat = F.normalize(self.mem_feat, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels):
        """
        Update features and corresponding pseudo labels
        """
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # pseudo_labels = concat_all_gather(pseudo_labels)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).to(device) % self.K
        # aa2 = np.array(self.mem_labels.cpu())
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        # aa=np.array(self.mem_labels.cpu())
        self.queue_ptr = end % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            feats_q: <B, D> query image features before normalization
            logits_q: <B, C> logits for class prediction from queries
            logits_ins: <B, K> logits for instance prediction
            k: <B, D> contrastive keys
        """

        # compute query features
        logits_q, feats_q = self.src_model(im_q)

        if cls_only:
            return logits_q,feats_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, k = self.momentum_model(im_k)
            k = F.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k
def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    return ema_model
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)
def train_one_epoch(config, src_model, data_loader, val_index=''):
    num_steps = len(data_loader.dataset)
    queue_size = num_steps
    momentum_model = deepcopy(src_model)
    model = AdaMoCo(
        src_model,
        momentum_model,
        K=queue_size
    ).to(device)
    optimizer = build_optimizer(config, model)

    img_shape = (512, 512, 3)
    n_pixels = img_shape[0]
    ap = 0.92
    mt_alpha = 0.999
    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5
    rst_m = 0.01
    tta_transforms = pth_transforms.Compose([
        # Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.6, 1.4],
            contrast=[0.7, 1.3],
            saturation=[0.5, 1.5],
            hue=[-0.06, 0.06],
            gamma=[0.7, 1.3]
        ),
        pth_transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        pth_transforms.RandomAffine(
            degrees=[-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.9, 1.1),
            shear=None
        ),
        pth_transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
        pth_transforms.CenterCrop(size=n_pixels),
        pth_transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, 0.005),
        # Clip(clip_min, clip_max)
    ])
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    all_outputs = []
    all_targets = []
    data_results = []
    banks = eval_and_label_dataset(data_loader, model, queue_size)
    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        loss_classifier_meter = AverageMeter()
        loss_entropy_meter = AverageMeter()
        loss_self_meter = AverageMeter()

        for idx, (images, masks, targetss, img_names) in enumerate(data_loader):
            # print('idx',idx)
            images_w = (images * masks).to(device)
            targetss=targetss.to(device)
            images_q=tta_transforms(images_w)
            images_k=tta_transforms(images_w)
            logits_w, feats_w = model(images_w, cls_only=True)
            # feats_w=F.normalize(feats_w, dim=1)
            with torch.no_grad():
                pseudo_labels_w, probs_w, _ = refine_predictions(
                    feats_w, banks
                )

            _, logits_q, logits_ins, keys = model(images_q, images_k)
            # update key features and corresponding pseudo labels
            model.update_memory(keys, pseudo_labels_w)
            # mem_labels=torch.argmax(banks['probs'])
            # moco instance discrimination
            # loss_ins = instance_loss(
            #     logits_ins=logits_ins,
            #     pseudo_labels=pseudo_labels_w,
            #     mem_labels=mem_labels,
            # )
            loss_ins = instance_loss(
                logits_ins=logits_ins,
                pseudo_labels=pseudo_labels_w,
                mem_labels=model.mem_labels,
            )

            # classification
            loss_cls = F.cross_entropy(logits_q, pseudo_labels_w)

            # diversification
            loss_div = div(logits_q)

            loss = loss_cls+ loss_ins+ loss_div
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # use slow feature to update neighbor space
            with torch.no_grad():
                logits_w, feats_w = model.momentum_model(images_w)
            # feats_w=F.normalize(feats_w, dim=1)
            banks=update_labels(banks, feats_w, logits_w)


            loss_classifier_meter.update(loss_cls.item(), images.size(0))
            loss_self_meter.update(loss_ins.item(), images.size(0))
            loss_entropy_meter.update(loss_div.item(), images.size(0))
            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'classifier_loss {loss_classifier_meter.val:.4f} ({loss_classifier_meter.avg:.4f})\t'
                    f'entropy_loss {loss_entropy_meter.val:.4f} ({loss_entropy_meter.avg:.4f})\t'
                    f'self_loss {loss_self_meter.val:.4f} ({loss_self_meter.avg:.4f})\t'
                )
            # logger.info(
        #     f'lr: {args.base_lr} {valdata} {ptnames}' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)
        # validate_network(data_loader, model, banks, val_index=val_index)
    return model,banks






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


if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)

    # model_name = ['pneullm',]
    model_name = ['convnext', ]
    # model_name = ['dinov3']
    # model_name = ['qwen2_5vision']
    # model_name = ['convnext']
    lrs=[5e-5,8e-5,1e-4, 1e-5, 3e-5]
    for kk in range(len(lrs)):
        args = parse_option()
        args.model_name = model_name[0]
        config = get_config(args)
        config.TRAIN.EPOCHS = 5
        config.TRAIN.WARMUP_EPOCHS = 0
        config.DATA.BATCH_SIZE = 10
        config.BASE_LR = lrs[kk]

        # pt_path='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/trainable_bestvalpt.pth'
        # pt_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
        pt_path ='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-b/batch16_lr1e-05_testall/model_best_acc_shanxi_test.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/2models_distill/dinov3stu224_trained_convnext_slgmstea/224_batch10_lr1e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout/3models_distill/dinov3st224_notrained_convnext_slgms_pneullmtea_0.5weight/224_batch10_lr3e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_3B/best_224_batch16_lr1e-05_testall/model_best_acc_shanxi_val.pth'
        config.OUTPUT = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/Adacontrast/'
        folder_path = config.OUTPUT
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = 'Adacontrast_result.txt'
        file_path = os.path.join(folder_path, file_name)
        logger_path = os.path.join(config.OUTPUT,
                                   f'{config.DATA.BATCH_SIZE}batch_{config.BASE_LR}lr')
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        if not os.path.exists(logger_path + '/analyze'):
            os.makedirs(logger_path + '/analyze')
        logger = create_logger(output_dir=logger_path, name=f"{config.MODEL.NAME}")

        path = os.path.join(logger_path, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot, total_inference_time=main(config, pt_path, logger_path)
        # avg_test_acc_org = (shanxi_test_org[1] * 3847 + guizhou_test_org[1] * 3144) / 6991
        # avg_test_acc_tent = (shanxi_test_shot[1] * 3847 + guizhoutest_shot[1] * 3144) / 6991

        with open(file_path, 'a', encoding='utf-8') as file:
            re = f'{args.model_name} {config.BASE_LR}:\n' \
                 f'ORG global_shanxi_test_avg: {shanxi_test_org[0]}, shanxi_test_auc: {shanxi_test_org[-1]}, shanxi_test_acc: {shanxi_test_org[1]}, shanxi_test_sensitivity: {shanxi_test_org[2]}, shanxi_test_specificity: {shanxi_test_org[3]}\n' \
                 f'ORG global_guizhou_test_avg: {guizhou_test_org[0]}, guizhou_test_auc: {guizhou_test_org[-1]}, guizhou_test_acc: {guizhou_test_org[1]}, guizhou_test_sensitivity: {guizhou_test_org[2]}, guizhou_test_specificity: {guizhou_test_org[3]}\n' \
                 f'total_inference_time: {total_inference_time} seconds\n' \
                 f'Adacontrast global_shanxi_test_avg: {shanxi_test_shot[0]}, shanxi_test_auc: {shanxi_test_shot[-1]}, shanxi_test_acc: {shanxi_test_shot[1]}, shanxi_test_sensitivity: {shanxi_test_shot[2]}, shanxi_test_specificity: {shanxi_test_shot[3]}\n' \
                 f'Adacontrast global_guizhou_test_avg: {guizhoutest_shot[0]}, guizhou_test_auc: {guizhoutest_shot[-1]}, guizhou_test_acc: {guizhoutest_shot[1]}, guizhou_test_sensitivity: {guizhoutest_shot[2]}, guizhou_test_specificity: {guizhoutest_shot[3]}\n\n'

            file.write(re)
# f'ORG global_all_test_acc: {avg_test_acc_org} ---> Adacontrast global_all_test_acc: {avg_test_acc_tent}\n' \

