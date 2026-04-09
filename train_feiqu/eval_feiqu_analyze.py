
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
    parser.add_argument('--batch-size', type=int, default=10, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls/224/contra_20251127/', type=str, metavar='PATH',
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
    parser.add_argument('--base_lr', type=float, default=1e-4)

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
def build_phe_loader(args):
    # 设置随机数种子
    setup_seed(args.seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset_train2 = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig', args.data_root + '/subregionsmasksbig',
                                               txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt',
                                               data_transform=transform,
                                               )
    dataset_val = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig', args.data_root + '/subregionsmasksbig',
                                               txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val_subregions.txt',
                                               data_transform=transform,
                                               )

    dataset_test = Shanxi_wmask_feiqu_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/subregionsimgsbig', '/disk3/wjr/dataset/nejm/guizhoudataset/subregionsmasksbig',
                                   txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one_subregions.txt',
                                   data_transform=transform,
                                   )
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    global_transfo_subregions = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_test_global = Shanxi_Submasks_Subregions_Dataset(args.data_root + '/subregionsimgsbig',
                                                      args.data_root + '/subregionsmasksbig',
                                                      txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                                      data_subregions_transform=global_transfo_subregions,
                                                      )
    dataset_test2_global = Shanxi_Submasks_Subregions_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/subregionsimgsbig',
                                                       '/disk3/wjr/dataset/nejm/guizhoudataset/subregionsmasksbig',
                                                       txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                                       data_subregions_transform=global_transfo_subregions)

    data_loader_test_global = torch.utils.data.DataLoader(
        dataset_test_global,
        batch_size=10,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2_global = torch.utils.data.DataLoader(
        dataset_test2_global,
        batch_size=10,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    return data_loader_train2, data_loader_val, data_loader_test, data_loader_test_global, data_loader_test2_global

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
from model_cls.convnext import *
from model_cls.ViP.ViP import ViP
from model_cls.ViP.CoCoOp import CoCoOp
from model_cls.pneullm.mm_adaptation import phellm, phellm_vision
from model_cls.dinov3.vit_dinov3 import *
from timm.utils import ModelEma as ModelEma
from model_cls.rad_dino.raddino import rad_dino_gh
from model_cls.vmambav2.vmamba import *
# from model_cls.qwen2_5vl import qwen2_5vision
from torch import optim as optim
def main(config, ptname='model_ema_best_auc_shanxi_val'):
    data_loader_train2, data_loader_val, data_loader_test, data_loader_test_global, data_loader_test2_global = build_phe_loader(args)
    logger.info(f"Creating model:{args.model_name}")
    if args.model_name=='resnet50':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        model = get_model('resnet50_8xb32_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        # net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.head.fc = nn.Linear(2048, 4)
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
                         sub_class=0,
                         main_class=5
                         )
        pretrained = '/disk3/wjr/workspace/sec_proj4/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        state_dict = torch.load(pretrained, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

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
    elif args.model_name == 'pneullm_vision':
        model=phellm_vision()
    elif args.model_name == 'hrnet18':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w18_3rdparty_8xb32-ssld_in1k_20220120-455f69ea.pth'
        model = get_model('hrnet-w18_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == 'hrnet48':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth'
        model = get_model('hrnet-w48_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(2048, 2)

    elif args.model_name == "convnext":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(768, 5)


        # model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
        # model.head.fc = nn.Linear(768, 2)
        # args.save_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/analyze'
        # ckp_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
        # if os.path.isfile(ckp_path):
        #     print(ckp_path)
        #     checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
        #     model.load_state_dict(checkpoint['model'], strict=True)

        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 5)
        # model = convnexttiny_org(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "convnext_dinov3":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_dino3(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
        model = convnextbase_dino3(pretrained=True, drop_path_rate=0.)
        model.head.fc = nn.Linear(1024, 4)
        # model = convnextlarge_dino3(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "dinov2":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'
        model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=pretrained_cfg)
        model.backbone.ln1 = nn.Linear(384, 4, bias=True)
    elif args.model_name == 'vim':
        model = build_model(config)
        model.head = nn.Linear(192, 4)
    elif args.model_name == 'PKA2_Net':
        model = PKA2_Net(n_class=4)
    elif args.model_name == 'pcam':
        model = pcam_dense121(num_classes=[1, 1,1,1])
        model.to(device)
    elif args.model_name == "swintiny":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(768, 5, bias=True)
    elif args.model_name == "vit-base":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
        model = get_model('vit-base-p16_in21k-pre_3rdparty_in1k-384px', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.layers.head = nn.Linear(768, 4, bias=True)
    elif args.model_name == "vit-large":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth'
        model = get_model('vit-large-p16_in21k-pre_3rdparty_in1k-384px', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.layers.head = nn.Linear(1024, 4, bias=True)
    elif args.model_name == 'pvt':
        model = pvt_tiny(pretrained=True)
        model.head = nn.Linear(512, 4)
    elif args.model_name == 'dmalnet':
        args.output_dir = '/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/DMALNet/_lr0.0003/models_params/RR_avgpool/'
        model = fusion_net(pretrained_pth=args.output_dir)
    model.to(device)
    ckp_path = os.path.join(config.OUTPUT, ptname + '.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
    shanxi_train_accuracy, shanxi_train_auc, shanxi_train_peracc, shanxi_train_perauc,shanxi_train_threshold  = validate_final(
        config, data_loader_train2, model,
         valdata='shanxi_train')

    validate_final(config, data_loader_val, model, valdata='shanxi_val', threshold=shanxi_train_threshold)

    validate_final(config, data_loader_test, model, valdata='guizhou_test', threshold=shanxi_train_threshold)

    # shanxi_val_accuracy_global, shanxi_val_auc_global, shanxi_val_sens_global, shanxi_val_spec_global = validate_final_global(
    #     config, data_loader_test_global, model,
    #     threshold=0.5, valdata='shanxi_test')
    #
    # guizhou_test_accuracy_global, guizhou_test_auc_global, guizhou_test_sens_global, guizhou_test_spec_global = validate_final_global(
    #     config, data_loader_test2_global, model,
    #     threshold=0.5, valdata='guizhou_test')
    #
    # return shanxi_train_auc, shanxi_train_accuracy, shanxi_train_perauc, shanxi_train_peracc, shanxi_val_auc, shanxi_val_accuracy, shanxi_val_perauc, shanxi_val_peracc, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_perauc, guizhou_test_peracc, shanxi_val_auc_global, shanxi_val_accuracy_global, shanxi_val_spec_global, shanxi_val_sens_global, guizhou_test_auc_global, guizhou_test_accuracy_global, guizhou_test_spec_global, guizhou_test_sens_global



def adaptive_calculate_metrics_global(all_outputs, all_targets_onehot, threshold=None, valdata='shanxi_test', num_classes=2,
                               plot_roc=True, save_fpr_tpr=True):
    probabilities = all_outputs
    all_targets = all_targets_onehot
    # predictions = (probabilities >= 0.5).astype(int)  # 设定阈值为 0.5
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
        auc_scores.append(auc)
        # 使用最优阈值计算预测值
        y_pred = (probabilities[:, i] >= threshold).astype(int)
        # y_pred = (probabilities[:, i] >= 0.5).astype(int)
        # 计算各项指标
        acc_scores.append(accuracy_score(all_targets[:, i], y_pred))
        f1_scores.append(f1_score(all_targets[:, i], y_pred))
        precision_scores.append(precision_score(all_targets[:, i], y_pred))
        recall_scores.append(recall_score(all_targets[:, i], y_pred))
        # 计算 Sensitivity 和 Specificity
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], y_pred).ravel()
        sensitivity_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

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
    local_probs=1-local_probs
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
def validate_final_global(config, data_loader, model,threshold=None, valdata='shanxi_test'):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, _) in enumerate(data_loader):
        # rand_module.randomize()
        # images = images.to(device)
        images = (images * masks).to(device).reshape(-1, 3, 224, 224)
        # images = images - F.interpolate(
        #     F.interpolate(images, scale_factor=0.5, mode='nearest', recompute_scale_factor=True),
        #     scale_factor=1 / 0.5, mode='nearest', recompute_scale_factor=True)
        # images = images * 2.0 / 3.0
        labels = targets.to(device)
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred = model(images)
        if args.model_name == "dinov2":
            pred = pred[1]
        prob = torch.nn.functional.softmax(pred, dim=1)
        prob=prob.reshape(masks.shape[0], masks.shape[1],5)
        subregions_outputs2 = torch.zeros((targets.shape[0], 6, 2))
        subregions_outputs2[:, :, 0] = prob[..., 0] + prob[..., 1]
        subregions_outputs2[:, :, 1] = prob[..., 2] + prob[..., 3] + prob[..., 4]
        global_prob = local2global_prob(subregions_outputs2)

        # 收集输出和目标
        all_outputs.extend(global_prob.float().data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    # all_outputs = torch.cat(all_outputs)
    # all_targets = torch.cat(all_targets)
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    # 计算整体指标
    # 计算整体指标
    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics_global(all_outputs, all_targets)

    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    print(
        f'{valdata} Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)
    return acc_local, auc_local, sensitivity_local, specificity_local

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score
import numpy as np
import torch

def calculate_auc_metrics(all_outputs, all_targets_onehot):
    # probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    probabilities = torch.sigmoid(all_outputs).cpu().numpy()
    all_targets = all_targets_onehot.cpu().numpy()
    auc = roc_auc_score(all_targets[:, 0], probabilities[:, 0])
    return auc * 100

from collections import defaultdict, Counter
@torch.no_grad()
def validate_final(config, data_loader, model,valdata='shanxi_test', threshold=None):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = defaultdict(list)
    all_targets = defaultdict(list)
    regions = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']
    for idx, (images, masks, targets, imagename) in enumerate(data_loader):
        # rand_module.randomize()
        # images = images.to(device)
        images = (images * masks).to(device)
        # images = images - F.interpolate(
        #     F.interpolate(images, scale_factor=0.5, mode='nearest', recompute_scale_factor=True),
        #     scale_factor=1 / 0.5, mode='nearest', recompute_scale_factor=True)
        # images = images * 2.0 / 3.0
        labels = targets.to(device).long()
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        subregions_outputs = model(images)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
        for ij in range(targets.shape[0]):
            imagenameij=imagename[ij]
            for region in regions:
                if region in imagenameij:
                    name=region
                    break
            # subregions_01=torch.zeros((1,2))
            # labels_01=labels[ij,...].unsqueeze(0).clone()
            # subregions_01[0,0]=subregions_outputs[ij,0]+subregions_outputs[ij,1]
            # subregions_01[0, 1] = subregions_outputs[ij, 2] + subregions_outputs[ij, 3] + subregions_outputs[ij, 4]
            # labels_01[labels_01==1]=0
            # labels_01[labels_01 >1] = 1
            # all_outputs[name].extend(subregions_01.float().data.cpu().numpy())
            # all_targets[name].extend(labels_01.cpu().numpy())

            all_outputs[name].extend(subregions_outputs[ij,...].unsqueeze(0).float().data.cpu().numpy())
            all_targets[name].extend(labels[ij,...].unsqueeze(0).cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    for region in regions:
        all_outputs_r = np.array(all_outputs[region])
        all_targets_r = np.array(all_targets[region])
        # 计算整体指标
        # 计算整体指标
        accuracy, auc, per_acc, per_auc, threshold = adaptive_calculate_metrics(all_outputs_r, all_targets_r, valdata,
                                                                                threshold)
        # 打印结果
        print(f'{region}: Accuracy: {accuracy:.4f}, AUC: {auc:.4f}', 'per_acc:', per_acc, 'per_auc:', per_auc)

    return accuracy, auc, per_acc, per_auc, threshold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve


def adaptive_calculate_metrics(all_outputs, all_targets, valdata, threshold=None):
    probabilities = all_outputs  # shape: (N, C)
    num_classes = probabilities.shape[-1]

    # ==============================
    # Step 1: 可视化概率分布（可选）
    # ==============================
    for i in range(num_classes):
        plt.hist(probabilities[all_targets == i, i], bins=50, alpha=0.5, label=f'Class {i}')
    plt.legend()
    save_path = config.OUTPUT + '/' + valdata + '_prob_hist.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

    # ==============================
    # Step 2: 默认预测（argmax）
    # ==============================
    preds_argmax = np.argmax(probabilities, axis=1)
    cm_argmax = confusion_matrix(all_targets, preds_argmax, labels=list(range(num_classes)))
    global_acc_argmax = np.mean(preds_argmax == all_targets)
    per_class_acc_argmax = cm_argmax.diagonal() / np.maximum(cm_argmax.sum(axis=1), 1)
    per_class_acc_argmax = np.nan_to_num(per_class_acc_argmax)
    acc_argmax = np.mean(per_class_acc_argmax)  # macro-acc

    # 保存混淆矩阵（argmax）
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_argmax, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")
    ax.set(xticks=np.arange(cm_argmax.shape[1]),
           yticks=np.arange(cm_argmax.shape[0]),
           xticklabels=range(num_classes),
           yticklabels=range(num_classes),
           xlabel='Predicted Label',
           ylabel='True Label')
    thresh = cm_argmax.max() / 2.
    for i in range(cm_argmax.shape[0]):
        for j in range(cm_argmax.shape[1]):
            ax.text(j, i, format(cm_argmax[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_argmax[i, j] > thresh else "black")
    ax.set_title('Confusion Matrix (Argmax)')
    plt.tight_layout()
    save_path = config.OUTPUT + '/' + valdata + '_confusion_map_argmax.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

    # ==============================
    # Step 3: 计算 AUC + 最优阈值
    # ==============================
    # ==============================
    # Step: 处理 threshold 参数
    # ==============================
    optimal_thresholds = []
    auc_scores = []

    if threshold is None:
        # 自动为每类计算最优阈值, 改用 Precision-Recall 曲线 + Max F1 阈值（强烈推荐）
        # Youden's J 基于 ROC，对不平衡数据不友好；PR 曲线更关注正类表现。
        for i in range(num_classes):
            binary_targets = (all_targets == i).astype(int)
            # 替换原来的 roc_curve 部分
            precision, recall, thresholds_pr = precision_recall_curve(binary_targets, probabilities[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_thresh = thresholds_pr[optimal_idx]  # 注意：thresholds_pr 长度 = len(precision)-1
            optimal_thresholds.append(optimal_thresh)
            # if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
            #     auc_scores.append(np.nan)
            #     optimal_thresholds.append(0.5)
            #     continue
            # fpr, tpr, thresholds_roc = roc_curve(binary_targets, probabilities[:, i])
            auc = roc_auc_score(binary_targets, probabilities[:, i])
            auc_scores.append(auc)
            # youden_j = tpr - fpr
            # optimal_idx = np.argmax(youden_j)
            # optimal_thresholds.append(thresholds_roc[optimal_idx])
    else:
        # 用户提供了 threshold
        if np.isscalar(threshold):
            # 单个值：广播到所有类
            optimal_thresholds = [float(threshold)] * num_classes
        else:
            # 假设是 list 或 array
            threshold = np.array(threshold)
            if threshold.shape[0] != num_classes:
                raise ValueError(f"Expected threshold length {num_classes}, got {threshold.shape[0]}")
            optimal_thresholds = threshold.tolist()
        # AUC 仍可计算（即使不用来定阈值）
        for i in range(num_classes):
            binary_targets = (all_targets == i).astype(int)
            if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
                auc_scores.append(np.nan)
            else:
                auc = roc_auc_score(binary_targets, probabilities[:, i])
                auc_scores.append(auc)

    auc_scores = np.array(auc_scores)
    mean_auc = np.mean(auc_scores[~np.isnan(auc_scores)])

    # ==============================
    # Step 4: 自适应阈值预测
    # ==============================
    final_preds = []
    for prob in probabilities:
        candidates = []
        for cls_idx, p in enumerate(prob):
            if p >= optimal_thresholds[cls_idx]:
                candidates.append((cls_idx, p))
        if candidates:
            final_pred = max(candidates, key=lambda x: x[1])[0]
        else:
            final_pred = np.argmax(prob)  # fallback
        final_preds.append(final_pred)
    final_preds = np.array(final_preds)

    # 计算新指标
    cm_thresh = confusion_matrix(all_targets, final_preds, labels=list(range(num_classes)))
    global_acc_thresh = np.mean(final_preds == all_targets)
    per_class_acc_thresh = cm_thresh.diagonal() / np.maximum(cm_thresh.sum(axis=1), 1)
    per_class_acc_thresh = np.nan_to_num(per_class_acc_thresh)
    acc_thresh = np.mean(per_class_acc_thresh)  # macro-acc

    # 保存混淆矩阵（阈值法）
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_thresh, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")
    ax.set(xticks=np.arange(cm_thresh.shape[1]),
           yticks=np.arange(cm_thresh.shape[0]),
           xticklabels=range(num_classes),
           yticklabels=range(num_classes),
           xlabel='Predicted Label',
           ylabel='True Label')
    thresh_val = cm_thresh.max() / 2.
    for i in range(cm_thresh.shape[0]):
        for j in range(cm_thresh.shape[1]):
            ax.text(j, i, format(cm_thresh[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_thresh[i, j] > thresh_val else "black")
    ax.set_title('Confusion Matrix (Adaptive Threshold)')
    plt.tight_layout()
    save_path = config.OUTPUT + '/' + valdata + '_confusion_map_thresh.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

    # ==============================
    # Step 5: 对比输出
    # ==============================
    print(f"\n=== Performance Comparison ({valdata}) ===")
    print(f"Method          | Global Acc (%) | Per Acc (%) | Mean AUC (%)")
    print(f"----------------|----------------|---------------|-------------")
    print(f"Argmax          | {global_acc_argmax * 100:14.2f} | {per_class_acc_argmax * 100} | {'-':12}")
    print(f"Adaptive Thresh | {global_acc_thresh * 100:14.2f} | {per_class_acc_thresh * 100} | {mean_auc * 100:12.2f}")
    print(f"Optimal thresholds: {optimal_thresholds}")

    return global_acc_thresh * 100, mean_auc * 100, per_class_acc_thresh * 100, auc_scores * 100, optimal_thresholds
if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    # lrs = [5e-5]
    lrs = [8e-5]
    # lrs = [5e-5]
    batch_sizes = [60]
    # model_name = ['hrnet18','resnet50', 'vit-base','swintiny']
    model_name = ['convnext']
    aug_rot=[True]
    aug_gaussian=[True]
    loss_type=['ce']
    pt_names=['model_best_auc_shanxi_train']
    for kk in range(len(lrs)):
        for qq in range(len(aug_gaussian)):
            for pp in range(len(aug_rot)):
                for jj in range(len(model_name)):
                    avgs = []
                    lr = lrs[kk]
                    args = parse_option()
                    args.batch_size = batch_sizes[0]
                    args.base_lr = lr
                    args.experi = 1
                    args.model_name = model_name[jj]
                    args.loss_type = loss_type[0]
                    args.aug_rot=aug_rot[pp]
                    args.aug_gaussian=aug_gaussian[qq]
                    config = get_config(args)
                    config.TRAIN.EPOCHS = 80
                    args.output='/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls_2global/224/contra_20251127/'
                    config.OUTPUT = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls_2global/224/contra_20251127/wmask_subcls_nodropout/convnext_databig_allsick_health/224_batch60_lr0.0001_testall'

                    config.freeze()

                    os.makedirs(config.OUTPUT, exist_ok=True)
                    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

                    path = os.path.join(config.OUTPUT, "config.json")
                    with open(path, "w") as f:
                        f.write(config.dump())
                    logger.info(f"Full config saved to {path}")

                    # print config
                    logger.info(config.dump())
                    logger.info(json.dumps(vars(args)))
                    for pt_name in pt_names:
                        main(config, pt_name)
                        gc.collect()
                        # shanxi_train_auc, shanxi_train_acc, shanxi_train_perauc, shanxi_train_peracc, shanxi_val_auc, shanxi_val_acc, shanxi_val_perauc, shanxi_val_peracc, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_perauc, guizhou_test_peracc, shanxi_test_auc_global, shanxi_test_accuracy_global, shanxi_test_sensitivity_global, shanxi_test_specificity_global, guizhou_test_auc_global, guizhou_test_accuracy_global, guizhou_test_sensitivity_global, guizhou_test_specificity_global = main(
                        #     config, pt_name)
                        # shanxi_avgs = (
                        #                       shanxi_test_auc_global + shanxi_test_accuracy_global + shanxi_test_sensitivity_global + shanxi_test_specificity_global) / 4
                        #
                        # guizhou_avgs = (
                        #                        guizhou_test_auc_global + guizhou_test_accuracy_global + guizhou_test_sensitivity_global + guizhou_test_specificity_global) / 4
                        #
                        # avg_test_acc = (shanxi_test_accuracy_global * 3847 + guizhou_test_accuracy_global * 3144) / 6991
                        # # 移除之前添加的处理器
                        # for handler in logger.handlers[:]:  # 循环遍历处理器的副本
                        #     logger.removeHandler(handler)  # 移除处理器
                        # gc.collect()
                        # folder_path = config.OUTPUT.split(config.OUTPUT.split('/')[-1])[0]
                        # if not os.path.exists(folder_path):
                        #     os.makedirs(folder_path)
                        # file_name = 'result.txt'
                        # file_path = os.path.join(folder_path, file_name)
                        # with open(file_path, 'a', encoding='utf-8') as file:
                        #     re = f'{args.model_name} {args.base_lr} {args.aug_rot}_augrot {args.aug_gaussian}_aug_gaussian: threshold: 0.5, {pt_name}:\n' \
                        #          f'local_shanxi_train_auc: {shanxi_train_auc}, shanxi_train_acc: {shanxi_train_acc}, shanxi_train_perauc: {shanxi_train_perauc}, shanxi_train_peracc: {shanxi_train_peracc}\n' \
                        #          f'local_shanxi_val_auc: {shanxi_val_auc}, shanxi_val_acc: {shanxi_val_acc}, shanxi_val_perauc: {shanxi_val_perauc}, shanxi_val_peracc: {shanxi_val_peracc}\n' \
                        #          f'local_guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_perauc: {guizhou_test_perauc}, guizhou_test_peracc: {guizhou_test_peracc}\n'\
                        #          f'global_all_test_acc: {avg_test_acc}\n' \
                        #          f'global_shanxi_test_avg: {shanxi_avgs}, shanxi_test_auc: {shanxi_test_auc_global}, shanxi_test_acc: {shanxi_test_accuracy_global}, shanxi_test_sensitivity: {shanxi_test_sensitivity_global}, shanxi_test_specificity: {shanxi_test_specificity_global}\n' \
                        #          f'global_guizhou_test_avg: {guizhou_avgs}, guizhou_test_auc: {guizhou_test_auc_global}, guizhou_test_acc: {guizhou_test_accuracy_global}, guizhou_test_sensitivity: {guizhou_test_sensitivity_global}, guizhou_test_specificity: {guizhou_test_specificity_global}\n\n'
                        #     file.write(re)