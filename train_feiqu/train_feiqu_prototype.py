
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
    # 动态构建转换列表（避免条件不满足时变量未定义）
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    if args.aug_rot==True:
        transform_list.append(transforms.RandomRotation(degrees=(-15, 15)))
    transform_list.extend([
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
    ])
    if args.aug_gaussian==True:
        transform_list.append(GaussianBlur(0.3))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    global_transfo2 = transforms.Compose(transform_list)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset_train = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig', args.data_root + '/subregionsmasksbig',
                                   txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt',
                                   data_transform=global_transfo2,
                                   )
    dataset_train_prototype = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig', args.data_root + '/subregionsmasksbig',
                                                txtpath=args.data_root + 'chinese_biaozhun_subregions.txt',
                                                data_transform=transform,
                                                )
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
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_train_prototype = torch.utils.data.DataLoader(
        dataset_train_prototype,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
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
    return data_loader_train, data_loader_train_prototype, data_loader_train2, data_loader_val, data_loader_test

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
from model_cls.pneullm.mm_adaptation import phellm, phellm_vision
from model_cls.dinov3.vit_dinov3 import *
from timm.utils import ModelEma as ModelEma
from model_cls.rad_dino.raddino import rad_dino_gh
from model_cls.vmambav2.vmamba import *
# from model_cls.qwen2_5vl import qwen2_5vision
from torch import optim as optim
def main(config, ptname='model_ema_best_auc_shanxi_val'):
    data_loader_train, data_loader_train_prototype, data_loader_train2, data_loader_val, data_loader_test = build_phe_loader(args)
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
    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        device='cpu' if args.model_ema_force_cpu else '',
        resume='')
    print("Using EMA with decay = %.8f" % args.model_ema_decay)

    ckp_path = os.path.join(config.OUTPUT, ptname+'.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        shanxi_train_accuracy, shanxi_train_auc, shanxi_train_peracc, shanxi_train_perauc = validate_final(
            config, data_loader_train2, data_loader_train_prototype, model,
            threshold=0.5, valdata='shanxi_train')

        shanxi_val_accuracy, shanxi_val_auc, shanxi_val_peracc, shanxi_val_perauc = validate_final(
            config, data_loader_val, data_loader_train_prototype, model,
            threshold=0.5, valdata='shanxi_val')

        guizhou_test_accuracy, guizhou_test_auc, guizhou_test_peracc, guizhou_test_perauc = validate_final(
            config, data_loader_test, data_loader_train_prototype, model,
            threshold=0.5, valdata='guizhou_test')

        return shanxi_train_auc, shanxi_train_accuracy, shanxi_train_perauc, shanxi_train_peracc, shanxi_val_auc, shanxi_val_accuracy, shanxi_val_perauc, shanxi_val_peracc, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_perauc, guizhou_test_peracc


    model_without_ddp = model
    if args.model_name == 'ViP':
        opt_para = list(model.model.prompt_learner.parameters()) + list(model.model.attn.parameters())
        opt_para.append(model.model.learnable_token)
        optimizer = torch.optim.SGD(
            opt_para,
            lr=config.BASE_LR,
            momentum=0.9,
            weight_decay=0.0005,
            dampening=0,
            nesterov=False,
        )
    elif args.model_name == 'CoCoOp':
        opt_para = list(model.model.prompt_learner.parameters())
        optimizer = torch.optim.SGD(
            opt_para,
            lr=config.BASE_LR,
            momentum=0.9,
            weight_decay=0.0005,
            dampening=0,
            nesterov=False,
        )
        # optimizer = optim.AdamW(opt_para, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
        #                         lr=config.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        # optimizer = build_optimizer(config, model, is_vip=True)
        # for name, param in model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)

    else:
        optimizer = build_optimizer(config, model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_acc_avg = 0.0
    max_auc_avg = 0.0
    max_acc_ema_avg = 0.0
    max_auc_ema_avg = 0.0
    max_acc_ema_avg2 = 0.0
    max_auc_ema_avg3 = 0.0
    max_auc_ema_avg4 = 0.0
    if args.loss_type=='focal_loss':
        criterion = FocalLoss(num_classes=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(num_classes=2)
    test_avg=0
    logger.info("Start training")
    start_time = time.time()
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    best_epoch=0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.sampler.set_epoch(epoch)
        # train_one_epoch(config, model, rand_module, criterion, data_loader_train, optimizer, epoch, mixup_fn=None, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler, model_ema=model_ema)
        writer=train_one_epoch(config, model, criterion, data_loader_train, data_loader_train_prototype, optimizer, epoch, mixup_fn=None,
                        lr_scheduler=lr_scheduler, loss_scaler=loss_scaler,model_ema=model_ema,  writer=writer)
        # 保存当前RNG状态
        cpu_state, cuda_state = save_rng_states()
        # loss, auc, acc, writer = validate(data_loader_train2, model, epoch, writer, index='shanxi_train')
        # loss, acc_ema, auc_ema, writer = validate(
        #     data_loader_train2, model_ema.ema, epoch, writer,
        #     index='shanxi_train_ema')
        loss, auc, acc, writer = validate(data_loader_train2, data_loader_train_prototype, model,epoch, writer, index='shanxi_val')
        test_loss, test_auc,test_acc, writer = validate(data_loader_val, data_loader_train_prototype, model, epoch, writer, index='shanxi_test')
        # test_loss, test_auc_ema, test_acc_ema, writer = validate(data_loader_test, model_ema.ema, epoch, writer, index='shanxi_test_ema')
        loss_test2, auc_test2, test_acc2,writer = validate(data_loader_test, data_loader_train_prototype, model, epoch, writer, index='guizhou_test')
        # loss_test2, auc_test2_ema, test_acc2_ema, writer = validate(data_loader_test2, model_ema.ema, epoch, writer,
        #                                                     index='guizhou_test_ema')
        # avg_test_acc = (test_acc * 3847 + test_acc2 * 3144) / 6991
        # writer.add_scalar('all_test acc_threshold0.5', avg_test_acc, epoch)
        val_avg = acc
        logger.info(f"Task average of the network on the val images: {val_avg:.4f}")
        if acc > max_acc_avg:
            best_epoch = epoch
            max_acc_avg = max(max_acc_avg, acc)
            save_test_checkpoint(config, epoch, model, max_acc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_shanxi_train.pth')
        else:
            max_acc_avg = max(max_acc_avg, acc)


        if test_acc > max_acc_ema_avg:
            best_epoch = epoch
            max_acc_ema_avg = max(max_acc_ema_avg, test_acc)
            save_test_checkpoint(config, epoch, model, max_acc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_shanxi_test.pth')
        else:
            max_acc_ema_avg = max(max_acc_ema_avg, test_acc)


        if auc > max_auc_avg:
            best_epoch = epoch
            max_auc_avg = max(max_auc_avg, acc)
            save_test_checkpoint(config, epoch, model, max_auc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_auc_shanxi_train.pth')
        else:
            max_auc_avg = max(max_auc_avg, auc)

        if test_acc2 > max_auc_ema_avg:
            best_epoch = epoch
            max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
            save_test_checkpoint(config, epoch, model, max_auc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_guizhou_test.pth')
        else:
            max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
        logger.info(f'Best epoch: {best_epoch}'+' Max acc average: '+"%.4f" % max_acc_avg+ ' Test auc average: '+"%.4f" % max_auc_avg)
        restore_rng_states(cpu_state, cuda_state)
    ckp_path = os.path.join(config.OUTPUT, ptname+'.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
    shanxi_train_accuracy, shanxi_train_auc, shanxi_train_peracc, shanxi_train_perauc = validate_final(
        config, data_loader_train2, model,
        threshold=0.5, valdata='shanxi_train')

    shanxi_val_accuracy, shanxi_val_auc, shanxi_val_peracc, shanxi_val_perauc = validate_final(
        config, data_loader_val, model,
        threshold=0.5, valdata='shanxi_val')

    guizhou_test_accuracy, guizhou_test_auc, guizhou_test_peracc, guizhou_test_perauc = validate_final(
        config, data_loader_test, model,
        threshold=0.5, valdata='guizhou_test')

    return shanxi_train_auc, shanxi_train_accuracy, shanxi_train_perauc, shanxi_train_peracc, shanxi_val_auc, shanxi_val_accuracy, shanxi_val_perauc, shanxi_val_peracc, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_perauc, guizhou_test_peracc


class OrdinalMetricLoss(nn.Module):
    def __init__(self, pos_label="none", **kwargs):
        """
        Args:
            pos_label: str, optional
                How to define relationships between samples with the same label ("positive pairs").
                Must be one of {"none", "same", "lower", "upper", "both"}.
                    none: no loss is computed for positive pairs
                    same: target is set to 0.5 for positive pairs
                    lower: target is set to 0 for positive pairs
                    upper: target is set to 1 for positive pairs
                Defaults to "none".
        """
        super(OrdinalMetricLoss, self).__init__()
        self.pos_label = pos_label
        self.crit = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.crit2 = nn.SmoothL1Loss(beta=0.05, reduction='none')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.l1_criterion = nn.SmoothL1Loss(beta=0.05, reduction='mean')
        self.bcecrit = torch.nn.BCEWithLogitsLoss()
        # self.l1_criterion = nn.L1Loss()
        self.loss = torch.nn.BCELoss()

    @staticmethod
    def get_loss_names():
        return ["loss"]

    def forward(self, labelsorg, gradcammap_sick, masksorg, sub_index=False, masks_index=None):
        '''
                对于子肺区而言
                    loss_diff1: sick_cam+health_cam在肺部内的激活大于肺部外
                    loss_contra：患病肺区内，2health_cam<sick_cam，保证大于2/3的肺区为患病；反之健康肺区内，2health_cam>sick_cam，保证小于2/3的肺区为患病
                对于整体肺部而言：
                    loss_diff1: sick_cam+health_cam在肺部内的激活大于肺部外
                    # loss_health_contra: 健康胸片肺部mask内，2health_cam>sick_cam，保证小于2/3的患病肺部mask为患病
                    loss_contra_fore: 患病胸片的患病肺部mask内，2health_cam<sick_cam，保证大于2/3的患病肺部mask为患病
                    # loss_contra_back: 患病胸片的健康肺部mask内，2health_cam>sick_cam，保证小于2/3的患病肺部mask为患病
                    # loss_diff_health: 患病胸片内，健康肺部mask内health_cam>患病肺部mask内health_cam
                    # loss_diff_sick: 患病胸片内，患病肺部mask内sick_cam>健康肺部mask内sick_cam
        '''
        if isinstance(masksorg, list):
            scale_factor = masksorg[0].shape[-1] / gradcammap_sick.shape[-1]
            masks = masksorg[0]
        else:
            scale_factor = masksorg.shape[-1] / gradcammap_sick.shape[-1]
            masks = masksorg
        gradcammap_sick = F.interpolate(gradcammap_sick.unsqueeze(1), scale_factor=scale_factor, mode="bicubic")

        # scale_factor=gradcammap.shape[-1]/masks.shape[-1]
        # masks=F.interpolate(masks, scale_factor=scale_factor, mode="nearest").squeeze(1)
        masks_new = (masks > 0)
        masks_out = (masks == 0)
        # 肺区内部像素的梯度激活值按照排序算法去处理
        scores_fore = (gradcammap_sick * masks_new).sum(dim=-1).sum(dim=-1) / (masks_new.sum(dim=-1).sum(dim=-1) + 1e-5)
        scores_back = (gradcammap_sick * masks_out).sum(dim=-1).sum(dim=-1) / (masks_out.sum(dim=-1).sum(dim=-1) + 1e-5)

        if sub_index:
            loss_diff = torch.mean(torch.max(torch.zeros_like(scores_back), scores_back - 0.1))
            # loss_diff = torch.mean(torch.max(torch.zeros_like(scores_back), scores_back - scores_fore + 0.2))
            sick_scores_fore = (gradcammap_sick * masks_new).sum(dim=-1).sum(dim=-1) / (
                    masks_new.sum(dim=-1).sum(dim=-1) + 1e-5)
            diff_scores = torch.max(torch.zeros_like(sick_scores_fore),
                                    (1-labelsorg) * (-sick_scores_fore + 0.7)) + torch.max(
                torch.zeros_like(sick_scores_fore), labelsorg * (sick_scores_fore - 0.2))
            mask = (diff_scores != 0).detach()
            loss_contra = (mask * diff_scores).sum() / mask.sum()
            return loss_diff, loss_contra
        else:
            loss_diff1 = torch.mean(torch.max(torch.zeros_like(scores_back), scores_back - 0.1))
            maskss = []
            labelss = []
            for jj in range(masks.shape[0]):
                maskk = []
                labelkk = []
                for kk in range(6):
                    maskk.append(masksorg[kk + 1][jj, ...])
                    labelkk.append(labelsorg[kk + 1][jj, ...])
                maskk = torch.stack(maskk)
                labelkk = torch.stack(labelkk)
                maskss.append(maskk)
                labelss.append(labelkk)
            maskss = torch.stack(maskss)
            labelss = torch.stack(labelss)
            sick_scores_fore = (gradcammap_sick.unsqueeze(1) * maskss).sum(dim=-1).sum(dim=-1) / (
                    maskss.sum(dim=-1).sum(dim=-1) + 1e-5)
            diff_scores = torch.relu((1 - labelss) * (-sick_scores_fore + 0.7)) + torch.relu(
                labelss * (sick_scores_fore - 0.2))
            mask = (diff_scores != 0).detach()
            loss_contra = (mask * diff_scores).sum() / mask.sum()
            loss_diff = loss_diff1 + loss_contra
            return loss_diff, loss_diff1, loss_contra
def compute_cache_logits(image_features, cache, num_classes=5, alpha=2.0, beta=5.0, neg_mask_thresholds=(0.03,1.0)):
    """Compute logits using positive/negative cache."""
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item)
            cache_values.append(class_index)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.long), num_classes=num_classes).float().to(
        image_features.device)

    affinity = image_features @ cache_keys
    # aa=((-1) * (beta - beta * affinity)).exp()
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

def train_one_epoch(config, model, criterion, data_loader, data_loader_train_prototype, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, writer=None, pos_alpha=2, pos_beta=5):
    model.eval()
    with torch.no_grad():
        cache = {}
        for idx_prototype, (images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
                data_loader_train_prototype):
            samples = (images_prototype * masks_prototype[0]).to(device)
            targets = labels_prototype.long().to(device)
            features, patch_feas, output, _ = model(samples, is_training=True)
            features = F.normalize(features, dim=-1)
            for ij, item in enumerate(features):
                if targets[ij] in cache:
                    cache[targets[ij]].append(item.unsqueeze(0))
                else:
                    cache[targets[ij]] = [item.unsqueeze(0)]
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for idx, (images, masks, labels, imgnames) in enumerate(data_loader):
        # rand_module.randomize()
        samples = (images * masks[0]).to(device)
        # samples = images.to(device)
        # samples=samples-F.interpolate(F.interpolate(samples, scale_factor=0.5, mode='nearest', recompute_scale_factor=True),
        #               scale_factor=1 / 0.5, mode='nearest', recompute_scale_factor=True)
        # samples=samples*2.0/3.0

        # samples = images.to(device)
        targets=labels
        targets = targets.long().to(device)

        data_time.update(time.time() - end)

        # output= model(samples)
        features, patch_feas, output, _ = model(samples, is_training=True)
        features = F.normalize(features, dim=-1)
        output_prototype=output.clone()
        features_prototype=features.clone()
        output_prototype = output_prototype+compute_cache_logits(features_prototype, cache, output_prototype.shape[-1], pos_alpha, pos_beta)

        loss = criterion(output_prototype, targets)

        optimizer.zero_grad()
        lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss.backward()
        optimizer.step()

        # loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
        #                         parameters=model.parameters(), create_graph=is_second_order,
        #                         update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:

            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        # loss_map_meter.update(map_align_loss.item(), targets.size(0))
        # if grad_norm is not None:  # loss_scaler return None if not update
        #     norm_meter.update(grad_norm)

        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                )
            writer.add_scalar('Shanxitrain loss', loss.item(), epoch * num_steps + idx)
            # writer.add_scalar('Shanxitrain cam_contra_sick_health_loss', loss_contraii.item(), epoch * num_steps + idx)
        model.eval()
        with torch.no_grad():
            cache = {}
            for idx_prototype, (images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
                    data_loader_train_prototype):
                samples = (images_prototype * masks_prototype[0]).to(device)
                targets = labels_prototype.long().to(device)
                features, patch_feas, output, _ = model(samples, is_training=True)
                features = F.normalize(features, dim=-1)
                for ij, item in enumerate(features):
                    if targets[ij] in cache:
                        cache[targets[ij]].append(item.unsqueeze(0))
                    else:
                        cache[targets[ij]] = [item.unsqueeze(0)]
        if (idx+1)%50==0:
            model.eval()
            with torch.no_grad():
                cache = {}
                for idx_prototype, (
                images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
                        data_loader_train_prototype):
                    samples = (images_prototype * masks_prototype[0]).to(device)
                    targets = labels_prototype.long().to(device)
                    features, patch_feas, output, _ = model(samples, is_training=True)
                    features = F.normalize(features, dim=-1)
                    for ij, item in enumerate(features):
                        if targets[ij] in cache:
                            cache[targets[ij]].append(item.unsqueeze(0))
                        else:
                            cache[targets[ij]] = [item.unsqueeze(0)]
            model.train()
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return writer
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

@torch.no_grad()
def validate(data_loader, data_loader_train_prototype, model,epoch, writer, index='shanxi_val', pos_alpha=2, pos_beta=5):
    model.eval()
    cache = {}
    for idx_prototype, (images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
            data_loader_train_prototype):
        samples = (images_prototype * masks_prototype[0]).to(device)
        targets = labels_prototype.long().to(device)
        features, patch_feas, output, _ = model(samples, is_training=True)
        features = F.normalize(features, dim=-1)
        for ij, item in enumerate(features):
            if targets[ij] in cache:
                cache[targets[ij]].append(item.unsqueeze(0))
            else:
                cache[targets[ij]] = [item.unsqueeze(0)]
    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, _) in enumerate(data_loader):
        # rand_module.randomize()
        images = (images * masks).to(device)
        # images = images - F.interpolate(
        #     F.interpolate(images, scale_factor=0.5, mode='nearest', recompute_scale_factor=True),
        #     scale_factor=1 / 0.5, mode='nearest', recompute_scale_factor=True)
        # images = images * 2.0 / 3.0
        # images = images.to(device)
        labels = targets.to(device).long()
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):

        features, patch_feas, output, _ = model(images, is_training=True)
        features = F.normalize(features, dim=-1)
        output_prototype = output.clone()
        features_prototype = features.clone()
        pred = output_prototype + compute_cache_logits(features_prototype, cache,
                                                                   output_prototype.shape[-1], pos_alpha, pos_beta)


        if args.model_name == "dinov2":
            pred = pred[1]
        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.float().data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        loss = nn.CrossEntropyLoss()(pred, labels)

        loss_meter.update(loss.item(), labels.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    acc, auc, peracc, perauc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=index,
        plot_roc=True,
        save_fpr_tpr=True)
    writer.add_scalar(index + ' auc', auc, epoch)
    writer.add_scalar(index + ' auc_0_0', peracc[0], epoch)
    writer.add_scalar(index + ' auc_0_1', perauc[1], epoch)
    writer.add_scalar(index + ' auc_1_0', perauc[2], epoch)
    writer.add_scalar(index + ' auc_1_1', perauc[3], epoch)
    writer.add_scalar(index + ' auc_1+', perauc[4], epoch)
    writer.add_scalar(index + ' acc', acc, epoch)
    writer.add_scalar(index + ' acc_0_0', peracc[0], epoch)
    writer.add_scalar(index + ' acc_0_1', peracc[1], epoch)
    writer.add_scalar(index + ' acc_1_0', peracc[2], epoch)
    writer.add_scalar(index + ' acc_1_1', peracc[3], epoch)
    writer.add_scalar(index + ' acc_1+', peracc[4], epoch)

    # writer.add_scalar(index + ' threshold', threshold, epoch)
    logger.info(
        f'{index} epoch: {epoch}' + ' auc: ' + "%.4f" % auc + ' acc: ' + "%.4f" % acc)

    return loss_meter.avg, auc, acc, writer


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

    # plt.ylabel('True label', fontsize=label_font_size)
    # plt.xlabel('Predicted label', fontsize=label_font_size)

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
    save_path = config.OUTPUT + '/'+valdata+'_confusion_map.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距

    # # 显示图像
    # plt.show()
    #
    # # 自动关闭图像
    # plt.close()
def adaptive_calculate_metrics(all_outputs, all_targets,threshold=None, valdata='shanxi_test', plot_roc=True, save_fpr_tpr=True):

    probabilities=all_outputs
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
    per_class_aucs = []
    preds = np.argmax(all_outputs, axis=1)
    num_classes=all_outputs.shape[-1]
    # Step 3: 计算 per-class accuracy (== recall)
    cm = confusion_matrix(all_targets, preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)  # 避免除零
    per_acc = np.nan_to_num(per_class_acc)  # 若某类无样本，则 acc=0
    for i in range(all_outputs.shape[-1]):
        binary_targets = (all_targets == i).astype(int)
        auc = roc_auc_score(binary_targets, probabilities[:, i])
        auc_scores.append(auc)
    auc_scores = np.array(auc_scores)
    auc = np.mean(auc_scores[~np.isnan(auc_scores)])  # 计算有效 AUC 的平均值
    acc = np.mean(per_acc[~np.isnan(per_acc)])
    global_acc = np.mean(preds == all_targets)  # 最简单直接！
    return global_acc * 100, auc * 100, per_acc * 100, auc_scores * 100

@torch.no_grad()
def validate_final(config, data_loader, data_loader_train_prototype, model,threshold=None, valdata='shanxi_test', pos_alpha=2, pos_beta=5):
    model.eval()
    cache = {}
    for idx_prototype, (images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
            data_loader_train_prototype):
        samples = (images_prototype * masks_prototype[0]).to(device)
        targets = labels_prototype.long().to(device)
        features, patch_feas, output, _ = model(samples, is_training=True)
        features = F.normalize(features, dim=-1)
        for ij, item in enumerate(features):
            if targets[ij] in cache:
                cache[targets[ij]].append(item.unsqueeze(0))
            else:
                cache[targets[ij]] = [item.unsqueeze(0)]

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, _) in enumerate(data_loader):
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
        features, patch_feas, output, _ = model(images, is_training=True)
        features = F.normalize(features, dim=-1)
        output_prototype = output.clone()
        features_prototype = features.clone()
        pred = output_prototype + compute_cache_logits(features_prototype, cache,
                                                                   output_prototype.shape[-1], pos_alpha, pos_beta)



        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.float().data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # all_outputs.append(pred)
        # all_targets.append(labels)
        loss = nn.CrossEntropyLoss()(pred, labels)
        # loss = nn.BCEWithLogitsLoss()(pred, labels)

        loss_meter.update(loss.item(), labels.size(0))
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
    accuracy, auc, per_acc, per_auc = adaptive_calculate_metrics(all_outputs, all_targets)


    # 打印结果
    print(f'Accuracy: {accuracy:.4f}, AUC: {auc:.4f}', 'per_acc:', per_acc, 'per_auc:',per_auc)
    return accuracy, auc, per_acc, per_auc

if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    # lrs = [5e-5]
    lrs = [8e-5, 1e-4, 5e-5, 3e-5, 1e-5]
    # lrs = [5e-5]
    batch_sizes = [20]
    # model_name = ['hrnet18','resnet50', 'vit-base','swintiny']
    model_name = ['dinov3',]
    aug_rot=[True]
    aug_gaussian=[True]
    loss_type=['ce']
    pt_names=['model_best_acc_shanxi_train','model_best_auc_shanxi_train','model_best_acc_shanxi_test','model_best_acc_guizhou_test']
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
                    config.TRAIN.EPOCHS = 50
                    config.OUTPUT = os.path.join(args.output, 'wmask_subcls_nodropout',args.model_name+'_prototype',
                                                 f'224_batch{args.batch_size}_lr' + str(args.base_lr)+'_testall')

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
                        shanxi_train_auc, shanxi_train_acc, shanxi_train_perauc, shanxi_train_peracc, shanxi_val_auc, shanxi_val_acc, shanxi_val_perauc, shanxi_val_peracc, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_perauc, guizhou_test_peracc = main(
                            config, pt_name)
                        # shanxi_train_avgs = (
                        #                             shanxi_train_auc + shanxi_train_acc + shanxi_train_sens + shanxi_train_spec) / 4
                        #
                        # shanxi_val_avgs = (shanxi_val_auc + shanxi_val_acc + shanxi_val_sens + shanxi_val_spec) / 4
                        #
                        # guizhou_avgs = (
                        #                        guizhou_test_auc + guizhou_test_accuracy + guizhou_test_sensitivity + guizhou_test_specificity) / 4
                        #
                        # 移除之前添加的处理器
                        for handler in logger.handlers[:]:  # 循环遍历处理器的副本
                            logger.removeHandler(handler)  # 移除处理器
                        gc.collect()
                        folder_path = config.OUTPUT.split(config.OUTPUT.split('/')[-1])[0]
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        file_name = 'result.txt'
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, 'a', encoding='utf-8') as file:
                            re = f'{args.model_name} {args.base_lr}: {pt_name}:\n' \
                                 f'shanxi_train_auc: {shanxi_train_auc}, shanxi_train_acc: {shanxi_train_acc}, shanxi_train_perauc: {shanxi_train_perauc}, shanxi_train_peracc: {shanxi_train_peracc}\n' \
                                 f'shanxi_val_auc: {shanxi_val_auc}, shanxi_val_acc: {shanxi_val_acc}, shanxi_val_perauc: {shanxi_val_perauc}, shanxi_val_peracc: {shanxi_val_peracc}\n' \
                                 f'guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_perauc: {guizhou_test_perauc}, guizhou_test_peracc: {guizhou_test_peracc}\n\n'
                            file.write(re)




