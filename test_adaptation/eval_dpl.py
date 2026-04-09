'''DPL[TMM2025] Decoupled Prototype Learning for Reliable  Test-Time Adaptation'''
'''ETent: Fully testtime adaptation by entropy minimization, ICLR2021'''
'''COME: Test-time adaption by Conservatively Minimizing Entropy, ICLR2025 '''
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
sys.path.append('/disk3/wjr/workspace/sec_proj4/proj4_feiqu_baseline//')
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
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + '/seg_rec_mask_1024',
                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
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


        model = train_one_epoch(config, model, data_loader_test)

        save_state = {'model': model.state_dict()}
        save_path = os.path.join(logger_path, 'shanxi_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        shanxi_test_shot=validate_network(data_loader_test, model, val_index='shanxi_test', save_path=logger_path+'/analyze/')
        end_time = time.time()
        # 总耗时（包含数据加载）
        total_inference_time = end_time - start_time
        print(f"{args.model_name}: Total inference time (including data loading): {total_inference_time:.4f} seconds")
        logger.info(
            f"{args.model_name}: Total inference time (including data loading): {total_inference_time:.4f} seconds")

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
        shanxi_test_org = validate_network(data_loader_test, model, 'shanxi_test_org', save_path=logger_path+'/analyze/')
        guizhou_test_org = validate_network(data_loader_test2, model, 'guizhou_test_org', save_path=logger_path+'/analyze/')
        model = train_one_epoch(config, model, data_loader_test2)
        save_state = {'model': model.state_dict()}
        save_path = os.path.join(logger_path, 'guizhou_test.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        guizhoutest_shot=validate_network(data_loader_test2, model, val_index='guizhou_test', save_path=logger_path+'/analyze/')
        # obtain_label(data_loader_val, model,'shanxival')
        # obtain_label(data_loader_test, model, 'shanxitest')
        # obtain_label(data_loader_test2, model, 'guizhoutest')
        return shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot, total_inference_time


import operator


class MemoryBank:
    """记忆库：动量更新类别伪特征（论文公式5）"""

    def __init__(self, num_classes, feature_dim, classifier_head, momentum=0.99):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        # 初始化：伪特征 = 源模型分类器权重（论文初始化策略）
        self.pseudo_features = classifier_head  # [C, D]

    def update(self, features, pseudo_labels):
        """
        动量更新伪特征
        Args:
            features: 置信样本的特征 [N, D]
            pseudo_labels: 置信样本的伪标签 [N]
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (pseudo_labels == c)
                if mask.sum() == 0:
                    continue  # 该类别无置信样本，不更新
                # 计算该类别的特征均值
                class_features = features[mask].mean(dim=0)
                # 动量更新：z_k* = η*z_k* + (1-η)*均值
                self.pseudo_features[c] = self.momentum * self.pseudo_features[c] + (1 - self.momentum) * class_features
        return self.pseudo_features


class AdaIN(nn.Module):
    """自适应实例归一化（论文公式8-10）：用于非置信样本风格迁移"""

    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_feat):
        """
        风格迁移：将style_feat的风格迁移到content_feat
        Args:
            content_feat: 置信样本特征 [N, C, H, W]
            style_feat: 非置信样本特征 [N, C, H, W]
        Returns:
            stylized_feat: 风格迁移后的特征 [N, C, H, W]
        """
        # 计算content特征的均值和方差（公式8-9）
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)  # [N, C, 1, 1]
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-6  # 避免除零

        # 计算style特征的均值和方差
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)  # [N, C, 1, 1]
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-6

        # 归一化content + 重缩放+偏移（公式10）
        normalized = (content_feat - content_mean) / content_std
        stylized_feat = style_std * normalized + style_mean
        return stylized_feat


class DPL(nn.Module):
    """解耦原型学习（DPL）核心模块"""

    def __init__(self, model, feature_dim=768, num_classes=2,
                 temperature=0.1, momentum=0.99, beta=0.1, conf_threshold=0.8):
        super(DPL, self).__init__()
        self.model = model  # 特征提取器（如ResNet-50）
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature  # 温度参数（论文默认0.1）
        self.beta = beta  # 正则化项权重（论文公式7）
        self.conf_threshold = conf_threshold  # 伪标签置信度阈值
        classifier_weight=self.model.head.fc.weight
        # 初始化组件
        self.memory_bank = MemoryBank(num_classes, feature_dim, classifier_weight,momentum)
        self.adain = AdaIN()

    def get_pseudo_labels(self, logits):
        """生成伪标签和置信度（论文公式1）"""
        probs = F.softmax(logits, dim=1)
        confidences, pseudo_labels = probs.max(dim=1)
        return pseudo_labels, confidences

    def dpl_loss(self, prototypes, features, pseudo_labels):
        """
        核心DPL损失（论文公式4）：原型中心损失
        Args:
            prototypes: 类别原型（分类器权重）[C, D]
            features: 置信样本特征 [N, D]
            pseudo_labels: 置信样本伪标签 [N]
        Returns:
            loss: DPL损失值
        """
        # 计算特征与所有原型的余弦相似度
        sim = F.cosine_similarity(features.unsqueeze(1), prototypes.unsqueeze(0), dim=2)  # [N, C]
        sim = sim / self.temperature

        # 按类别分组计算损失（避免重复计算）
        loss = 0.0
        C_prime = 0  # 当前batch中存在的类别数
        for c in range(self.num_classes):
            mask = (pseudo_labels == c)
            if mask.sum() == 0:
                continue
            C_prime += 1
            # 该类别的正样本相似度
            pos_sim = sim[mask, c].unsqueeze(1)  # [N_c, 1]
            # 该类别的负样本相似度（所有其他类别）
            neg_sim = sim[mask, :c] if c > 0 else torch.tensor([], device=sim.device)
            if c < self.num_classes - 1:
                neg_sim = torch.cat([neg_sim, sim[mask, c + 1:]], dim=1)  # [N_c, C-1]
            # 计算该类别的损失
            class_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [N_c, C]
            class_loss = F.cross_entropy(class_sim, torch.zeros_like(mask[mask]).long())
            loss += class_loss

        return loss / C_prime if C_prime > 0 else torch.tensor(0.0, device=features.device)

    def reg_loss(self, prototypes, pseudo_features):
        """
        正则化损失（论文公式6）：使用记忆库伪特征
        Args:
            prototypes: 类别原型 [C, D]
            pseudo_features: 记忆库中的伪特征 [C, D]
        Returns:
            reg_loss: 正则化损失值
        """
        sim = F.cosine_similarity(pseudo_features.unsqueeze(1), prototypes.unsqueeze(0), dim=2)  # [C, C]
        sim = sim / self.temperature
        # 每个伪特征对应自身原型为正样本
        labels = torch.arange(self.num_classes, device=sim.device)
        reg_loss = F.cross_entropy(sim, labels)
        return reg_loss

    def loss_forward(self, x):
        """
        前向传播（双阶段：论文图3(b)）
        Args:
            x: 目标域batch [B, 3, H, W]
        Returns:
            logits: 最终预测 [B, C]
            total_loss: DPL总损失
        """
        # -------------------------- 第一阶段：生成伪标签，划分置信/非置信样本 --------------------------

        # 生成伪标签和置信度
        logits, feat_high, feat_low = self.model(x)  # [B, C]
        pseudo_labels, confidences = self.get_pseudo_labels(logits)

        # 划分置信样本（X^a）和非置信样本（X^b）
        conf_mask = (confidences >= self.conf_threshold)
        if not conf_mask.any():
            # 无置信样本时，返回原始预测和0损失
            return logits, torch.tensor(0.0, device=x.device)

        feat_high_conf = feat_high[conf_mask]  # [N_a, D]
        pseudo_labels_conf = pseudo_labels[conf_mask]  # [N_a]
        feat_low_conf = feat_low[conf_mask]  # [N_a, C_low, H_low, W_low]

        # -------------------------- 第二阶段：非置信样本风格迁移 --------------------------
        unconf_mask = ~conf_mask
        if unconf_mask.any():
            feat_low_unconf = feat_low[unconf_mask]  # [N_b, C_low, H_low, W_low]
            # 随机配对：置信样本 ↔ 非置信样本
            batch_size_conf = feat_low_conf.shape[0]
            batch_size_unconf = feat_low_unconf.shape[0]
            match_size = min(batch_size_conf, batch_size_unconf)

            # 随机采样配对索引
            conf_idx = torch.randperm(batch_size_conf)[:match_size]
            unconf_idx = torch.randperm(batch_size_unconf)[:match_size]

            # 风格迁移：将非置信样本的风格迁移到置信样本
            stylized_feat_low = self.adain(feat_low_conf[conf_idx],
                                           feat_low_unconf[unconf_idx])  # [M, C_low, H_low, W_low]

            # 提取风格迁移后的高层特征
            stylized_feat_high = stylized_feat_low.mean([-2, -1])  # [M, D]

            # 合并原始置信特征和风格迁移特征
            feat_high_combined = torch.cat([feat_high_conf, stylized_feat_high], dim=0)  # [N_a + M, D]
            pseudo_labels_combined = torch.cat([pseudo_labels_conf, pseudo_labels_conf[conf_idx]], dim=0)  # [N_a + M]
        else:
            # 无非置信样本时，直接使用置信样本
            feat_high_combined = feat_high_conf
            pseudo_labels_combined = pseudo_labels_conf

        # -------------------------- 第三阶段：计算DPL总损失 --------------------------
        # 1. 更新记忆库
        self.memory_bank.update(feat_high_conf, pseudo_labels_conf)
        pseudo_features = self.memory_bank.pseudo_features

        # 2. 原型 = 分类器权重（论文核心设定）
        prototypes = self.model.head.fc.weight # [C, D]

        # 3. 计算核心DPL损失（公式4）
        loss_dpl = self.dpl_loss(prototypes, feat_high_combined, pseudo_labels_combined)

        # 4. 计算正则化损失（公式6）
        loss_reg = self.reg_loss(prototypes, pseudo_features)

        # 5. 总损失（公式7）
        total_loss = loss_dpl + self.beta * loss_reg

        return logits, total_loss, loss_dpl, loss_reg
    def forward(self, x):
        """
        前向传播（双阶段：论文图3(b)）
        Args:
            x: 目标域batch [B, 3, H, W]
        Returns:
            logits: 最终预测 [B, C]
            total_loss: DPL总损失
        """
        # -------------------------- 第一阶段：生成伪标签，划分置信/非置信样本 --------------------------

        # 生成伪标签和置信度
        logits, feat_high, feat_low = self.model(x)  # [B, C]
        return logits, feat_high, feat_low

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
            imsave_path = save_path + '/' + valdata + '_roc_map.pdf'
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
def validate_network(val_loader, model, val_index='shanxi_val', save_path=''):
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
            output=output[0]
        prob = torch.nn.functional.softmax(output, dim=1).float()
        for bb in range(prob.shape[0]):
            result = {}
            result["Image Index"] = imgname[bb]
            result["Finding Labels"] = imgname[bb].split('_')[0]
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
    df_data_results = pd.DataFrame(data_results)
    df_data_results.to_csv(save_path + '/' + val_index + '_dataprob_result.csv', index=False)

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

def dirichlet_entropy(x: torch.Tensor):#key component of COME
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = 1000 / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def train_one_epoch(config, model, data_loader):
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    model=DPL(model)
    optimizer = build_optimizer(config, model)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader))
    # Step 1: 冻结整个模型
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: 只解冻 LayerNorm2d 的参数
    for name, module in model.named_modules():
        if 'LayerNorm' in module.__class__.__name__:
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfrozen LayerNorm2d: {name}")

    logger.info("Start training")

    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        loss_meter = AverageMeter()
        loss_entropy_meter = AverageMeter()
        loss_reg_meter = AverageMeter()
        for idx, (images, masks, targetss, img_names) in enumerate(data_loader):
            images = (images * masks).to(device)
            targetss=targetss.to(device)
            logits, loss, loss_dpl, loss_reg = model.loss_forward(images)

            # loss = classifier_loss * 0.3 + entropy_loss * 1.
            optimizer.zero_grad()
            # lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            loss_entropy_meter.update(loss_dpl.item(), images.size(0))
            loss_reg_meter.update(loss_reg.item(), images.size(0))
            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'dpl_loss {loss_entropy_meter.val:.4f} ({loss_entropy_meter.avg:.4f})\t'
                    f'reg_loss {loss_reg_meter.val:.4f} ({loss_reg_meter.avg:.4f})\t'
                )
        # validate_network(data_loader, model, 'shanxi_test')
    # mem_label = obtain_label(data_loader, model)
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


if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)

    # model_name = ['pneullm',]
    # model_name = ['slgms', ]
    # model_name = ['dinov3']
    # model_name = ['qwen2_5vision']
    model_name = ['convnext']
    lrs=[5e-5, 1e-5, 3e-5,8e-5, 1e-4]
    for kk in range(len(lrs)):
        args = parse_option()
        args.model_name = model_name[0]
        config = get_config(args)
        config.TRAIN.EPOCHS = 5
        config.TRAIN.WARMUP_EPOCHS = 0
        config.DATA.BATCH_SIZE = 20
        config.BASE_LR = lrs[kk]

        # pt_path='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/trainable_bestvalpt.pth'
        # pt_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
        pt_path ='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-b/batch16_lr1e-05_testall/model_best_acc_shanxi_test.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/2models_distill/dinov3stu224_trained_convnext_slgmstea/224_batch10_lr1e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout/3models_distill/dinov3st224_notrained_convnext_slgms_pneullmtea_0.5weight/224_batch10_lr3e-05_testall/model_best_acc_alltest.pth'
        # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_3B/best_224_batch16_lr1e-05_testall/model_best_acc_shanxi_val.pth'
        config.OUTPUT = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/DPL_LN/'
        folder_path = config.OUTPUT
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = 'DPL_result.txt'
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
        shanxi_test_org, guizhou_test_org, shanxi_test_shot, guizhoutest_shot,total_inference_time=main(config, pt_path, logger_path)
        # avg_test_acc_org = (shanxi_test_org[1] * 3847 + guizhou_test_org[1] * 3144) / 6991
        # avg_test_acc_tent = (shanxi_test_shot[1] * 3847 + guizhoutest_shot[1] * 3144) / 6991

        with open(file_path, 'a', encoding='utf-8') as file:
            re = f'{args.model_name} {config.BASE_LR}:\n' \
                 f'ORG global_shanxi_test_avg: {shanxi_test_org[0]}, shanxi_test_auc: {shanxi_test_org[-1]}, shanxi_test_acc: {shanxi_test_org[1]}, shanxi_test_sensitivity: {shanxi_test_org[2]}, shanxi_test_specificity: {shanxi_test_org[3]}\n' \
                 f'ORG global_guizhou_test_avg: {guizhou_test_org[0]}, guizhou_test_auc: {guizhou_test_org[-1]}, guizhou_test_acc: {guizhou_test_org[1]}, guizhou_test_sensitivity: {guizhou_test_org[2]}, guizhou_test_specificity: {guizhou_test_org[3]}\n' \
                 f'total_inference_time: {total_inference_time} seconds\n' \
                 f'DPL global_shanxi_test_avg: {shanxi_test_shot[0]}, shanxi_test_auc: {shanxi_test_shot[-1]}, shanxi_test_acc: {shanxi_test_shot[1]}, shanxi_test_sensitivity: {shanxi_test_shot[2]}, shanxi_test_specificity: {shanxi_test_shot[3]}\n' \
                 f'DPL global_guizhou_test_avg: {guizhoutest_shot[0]}, guizhou_test_auc: {guizhoutest_shot[-1]}, guizhou_test_acc: {guizhoutest_shot[1]}, guizhou_test_sensitivity: {guizhoutest_shot[2]}, guizhou_test_specificity: {guizhoutest_shot[3]}\n\n'

            file.write(re)

# f'ORG global_all_test_acc: {avg_test_acc_org} ---> DPL global_all_test_acc: {avg_test_acc_tent}\n' \
