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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda:3')
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
    parser.add_argument('--output',
                        default='/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls_2global/224/contra_20251127/',
                        type=str, metavar='PATH',
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
# from torchvision import transforms as pth_transforms

def build_phe_loader(args):
    # 设置随机数种子
    setup_seed(args.seed)
    # 动态构建转换列表（避免条件不满足时变量未定义）
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    if args.aug_rot == True:
        transform_list.append(transforms.RandomRotation(degrees=(-15, 15)))
    transform_list.extend([
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
    ])
    if args.aug_gaussian == True:
        transform_list.append(GaussianBlur(0.3))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    global_transfo2 = transforms.Compose(transform_list)

    dataset_train = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig',
                                               args.data_root + '/subregionsmasksbig',
                                               txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt',
                                               data_transform=global_transfo2,
                                               )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset_train2 = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig',
                                                args.data_root + '/subregionsmasksbig',
                                                txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt',
                                                data_transform=transform,
                                                )
    dataset_prototype = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig',
                                             args.data_root + '/subregionsmasksbig',
                                             txtpath=args.data_root + 'chinese_ilo_book_subregions.txt',
                                             data_transform=transform,
                                             )
    dataset_val = Shanxi_wmask_feiqu_Dataset(args.data_root + '/subregionsimgsbig',
                                             args.data_root + '/subregionsmasksbig',
                                             txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val_subregions.txt',
                                             data_transform=transform,
                                             )

    dataset_test = Shanxi_wmask_feiqu_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/subregionsimgsbig',
                                              '/disk3/wjr/dataset/nejm/guizhoudataset/subregionsmasksbig',
                                              txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one_sick_subregions.txt',
                                              data_transform=transform,
                                              )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_prototype = torch.utils.data.DataLoader(
        dataset_prototype,
        batch_size=2,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )

    global_transfo_subregions = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val_global = Shanxi_Submasks_Subregions_Dataset(args.data_root + '/subregionsimgsbig',
                                                            args.data_root + '/subregionsmasksbig',
                                                            txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
                                                            data_subregions_transform=global_transfo_subregions,
                                                            )
    dataset_test_global = Shanxi_Submasks_Subregions_Dataset(args.data_root + '/subregionsimgsbig',
                                                             args.data_root + '/subregionsmasksbig',
                                                             txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                                             data_subregions_transform=global_transfo_subregions,
                                                             )
    dataset_test2_global = Shanxi_Submasks_Subregions_Dataset(
        '/disk3/wjr/dataset/nejm/guizhoudataset/subregionsimgsbig',
        '/disk3/wjr/dataset/nejm/guizhoudataset/subregionsmasksbig',
        txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
        data_subregions_transform=global_transfo_subregions)
    data_loader_val_global = torch.utils.data.DataLoader(
        dataset_val_global,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_test_global = torch.utils.data.DataLoader(
        dataset_test_global,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_test2_global = torch.utils.data.DataLoader(
        dataset_test2_global,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )

    return data_loader_train, data_loader_train2, data_loader_prototype, data_loader_val, data_loader_test, data_loader_val_global, data_loader_test_global, data_loader_test2_global


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
def main(config, ptname, pt_path):
    data_loader_train, data_loader_train2, data_loader_prototype, data_loader_val, data_loader_test, data_loader_val_global, data_loader_test_global, data_loader_test2_global = build_phe_loader(args)
    logger.info(f"Creating model:{args.model_name}")
    if args.model_name=='resnet50':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        model = get_model('resnet50_8xb32_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        # net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.head.fc = nn.Linear(2048, 5)
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
        model.head = nn.Linear(768, 5)

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
        # model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
        # model.head.fc = nn.Linear(768, 5)

        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 5)
        model = convnexttiny_org(pretrained=True, drop_path_rate=0.)
        model.head.fc = nn.Linear(768, 5)
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
    for param in model.parameters():
        param.requires_grad = False
    ckp_path = pt_path
    if os.path.isfile(ckp_path):
        state_dict = torch.load(ckp_path, map_location='cpu', weights_only=False)['model']
        state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=True)

        print('model_prototype')
        pos_cache=prototype_cache(model, data_loader_prototype, pos_shot_capacity=6)
        cache_keys = []
        cache_values = []
        for region in pos_cache:
            # 遍历该区域下每个类别
            for class_index in sorted(pos_cache[region].keys()):
                # 遍历该类别下的每个 item
                for item in pos_cache[region][class_index]:
                    cache_keys.append(item[0])
                    cache_values.append(class_index)
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0).to(device)
        cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.long), num_classes=5).float().to(device)
        model.adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device)
        model.adapter.weight = nn.Parameter(cache_keys.t())
        model.to(device)
        for param in model.adapter.parameters():
            param.requires_grad = True
        optimizer = build_optimizer(config, model.adapter)
        loss_scaler = NativeScalerWithGradNormCount()
        max_acc_avg = 0.0
        max_auc_avg = 0.0
        max_acc_ema_avg = 0.0
        max_auc_ema_avg = 0.0
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
        test_avg = 0
        logger.info("Start training")
        start_time = time.time()
        log_path = os.path.join(config.OUTPUT, 'logs')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)
        criterion = torch.nn.CrossEntropyLoss()

        best_epoch = 0
        for epoch in range(config.TRAIN.START_EPOCH, 30):
            writer = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch,lr_scheduler, cache_values, pos_alpha=2, pos_beta=5, writer=writer)
            # 保存当前RNG状态
            cpu_state, cuda_state = save_rng_states()
            auc, acc, writer =validate_final_global(data_loader_val_global, model, pos_alpha=2, pos_beta=5,
                                  valdata='shanxi_val', cache_values=cache_values, writer=writer, epoch=epoch)

            test_auc, test_acc, writer = validate_final_global(data_loader_test_global, model, pos_alpha=2, pos_beta=5,
                                  valdata='shanxi_test', cache_values=cache_values, writer=writer, epoch=epoch)
            auc_test2, test_acc2, writer = validate_final_global(data_loader_test2_global, model, pos_alpha=2, pos_beta=5,
                                  valdata='guizhou_test', cache_values=cache_values, writer=writer, epoch=epoch)


            val_avg = acc
            logger.info(f"Task average of the network on the val images: {val_avg:.4f}")
            if acc > max_acc_avg:
                best_epoch = epoch
                max_acc_avg = max(max_acc_avg, acc)
                save_test_checkpoint(config, epoch, model, max_acc_avg, optimizer, lr_scheduler,
                                     loss_scaler,
                                     logger, ptname='ftda_model_best_acc_shanxi_train.pth')
            else:
                max_acc_avg = max(max_acc_avg, acc)

            if test_acc > max_acc_ema_avg:
                best_epoch = epoch
                max_acc_ema_avg = max(max_acc_ema_avg, test_acc)
                save_test_checkpoint(config, epoch, model, max_acc_ema_avg, optimizer, lr_scheduler,
                                     loss_scaler,
                                     logger, ptname='ftda_model_best_acc_shanxi_test.pth')
            else:
                max_acc_ema_avg = max(max_acc_ema_avg, test_acc)

            if auc > max_auc_avg:
                best_epoch = epoch
                max_auc_avg = max(max_auc_avg, acc)
                save_test_checkpoint(config, epoch, model, max_auc_avg, optimizer, lr_scheduler,
                                     loss_scaler,
                                     logger, ptname='ftda_model_best_auc_shanxi_train.pth')
            else:
                max_auc_avg = max(max_auc_avg, auc)

            if test_acc2 > max_auc_ema_avg:
                best_epoch = epoch
                max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
                save_test_checkpoint(config, epoch, model, max_auc_ema_avg, optimizer, lr_scheduler,
                                     loss_scaler,
                                     logger, ptname='ftda_model_best_acc_guizhou_test.pth')
            else:
                max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
            logger.info(
                f'Best epoch: {best_epoch}' + ' Max acc average: ' + "%.4f" % max_acc_avg + ' Test auc average: ' + "%.4f" % max_auc_avg)
            restore_rng_states(cpu_state, cuda_state)
        ckp_path = os.path.join(config.OUTPUT, ptname + '.pth')
        if os.path.isfile(ckp_path):
            checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            auc, acc, writer = validate_final_global(data_loader_val_global, model, pos_alpha=2,
                                                     pos_beta=5,
                                                     valdata='shanxi_val', cache_values=cache_values, writer=writer,
                                                     epoch=0)

            test_auc, test_acc, writer = validate_final_global(data_loader_test_global, model,
                                                               pos_alpha=2, pos_beta=5,
                                                               valdata='shanxi_test', cache_values=cache_values,
                                                               writer=writer, epoch=0)
            auc_test2, test_acc2, writer = validate_final_global(data_loader_test2_global, model,
                                                                 pos_alpha=2, pos_beta=5,
                                                                 valdata='guizhou_test', cache_values=cache_values,
                                                                 writer=writer, epoch=0)



def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch,lr_scheduler, cache_values, pos_alpha=2, pos_beta=5, writer=None):
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
        targets = labels
        targets = targets.long().to(device)

        data_time.update(time.time() - end)
        _clsout, image_features = model(samples)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        affinity = model.adapter(image_features)
        cache_logits = ((-1) * (pos_beta - pos_beta * affinity)).exp() @ cache_values
        tip_logits = cache_logits+_clsout*pos_alpha
        loss = criterion(tip_logits, targets)
        optimizer.zero_grad()
        lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        loss_meter.update(loss.item(), targets.size(0))
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
        # torch.cuda.synchronize()
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return writer



def prototype_cache(model, data_loader_train_prototype, pos_shot_capacity=6):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    with torch.no_grad():
        pos_cache = defaultdict(lambda: defaultdict(list))

        for idx_prototype, (images_prototype, masks_prototype, labels_prototype, imgnames_prototype) in enumerate(
                data_loader_train_prototype):
            samples = (images_prototype * masks_prototype[0]).to(device)
            targets = labels_prototype.long().to(device)
            _clsout, image_features= model(samples)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # loss = criterion(_clsout, targets)
            loss = softmax_entropy(_clsout)
            for ij, item in enumerate(image_features):
                pos = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']
                for posii in pos:
                    if posii in imgnames_prototype[ij]:
                        name = posii
                update_cache(pos_cache[name], int(targets[ij]), [item.unsqueeze(0), loss[ij].unsqueeze(0)], pos_shot_capacity)
    return pos_cache

import operator


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
def validate_final_global(data_loader, model, cache_values, pos_alpha=2, pos_beta=5, valdata='shanxi_test', writer=None, epoch=0):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_outputs_org=[]
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
        _clsout, image_features= model(images)

        prob = torch.nn.functional.softmax(_clsout, dim=1).float()
        prob = prob.reshape(6, 5)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        affinity = model.adapter(image_features)
        cache_logits = ((-1) * (pos_beta - pos_beta * affinity)).exp() @ cache_values
        tip_logits = cache_logits + _clsout * pos_alpha
        prob = torch.nn.functional.softmax(tip_logits, dim=1).float()
        prob=prob.reshape(masks.shape[0], masks.shape[1],5)
        subregions_outputs2 = torch.zeros((targets.shape[0], 6, 2))
        subregions_outputs2[:, :, 0] = prob[..., 0] + prob[..., 1]
        subregions_outputs2[:, :, 1] = prob[..., 2] + prob[..., 3] + prob[..., 4]
        global_prob = local2global_prob(subregions_outputs2)
        all_outputs.extend(global_prob.float().data.cpu().numpy())

        proborg = torch.nn.functional.softmax(_clsout, dim=1).float()
        proborg = proborg.reshape(masks.shape[0], masks.shape[1], 5)
        subregions_outputs3 = torch.zeros((targets.shape[0], 6, 2))
        subregions_outputs3[:, :, 0] = proborg[..., 0] + proborg[..., 1]
        subregions_outputs3[:, :, 1] = proborg[..., 2] + proborg[..., 3] + proborg[..., 4]
        global_proborg = local2global_prob(subregions_outputs3)
        # 收集输出和目标
        all_outputs_org.extend(global_proborg.float().data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    # all_outputs = torch.cat(all_outputs)
    # all_targets = torch.cat(all_targets)
    all_outputs_org = np.array(all_outputs_org)
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    # 计算整体指标
    # 计算整体指标
    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics_global(
        all_outputs_org, all_targets)

    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    print(
        f'{valdata} ORG Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)


    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics_global(all_outputs, all_targets)

    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    print(
        f'{valdata} TDA Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)

    writer.add_scalar(valdata + ' auc', auc_local, epoch)
    writer.add_scalar(valdata + ' acc', acc_local, epoch)
    writer.add_scalar(valdata + ' specificity', specificity_local, epoch)
    writer.add_scalar(valdata + ' sensitivity', sensitivity_local, epoch)

    # writer.add_scalar(index + ' threshold', threshold, epoch)
    logger.info(
        f'{valdata} epoch: {epoch}' + ' auc: ' + "%.4f" % auc_local + ' acc: ' + "%.4f" % acc_local)

    return auc_local, acc_local, writer


def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]



def compute_cache_logits(image_features, cache, num_classes=5, alpha=2.0, beta=5.0, neg_mask_thresholds=(0.03,1.0)):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.long), num_classes=num_classes).float().to(image_features.device)

        affinity = image_features @ cache_keys
        # aa=((-1) * (beta - beta * affinity)).exp()
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

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

def adaptive_calculate_metrics(all_outputs, all_targets,valdata):

    probabilities=all_outputs
    # 初始化存储各项指标的列表
    auc_scores = []
    preds = np.argmax(all_outputs, axis=1)
    num_classes=all_outputs.shape[-1]
    # Step 3: 计算 per-class accuracy (== recall)
    cm = confusion_matrix(all_targets, preds, labels=list(range(num_classes)))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # # 添加颜色条
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")
    # # 设置坐标轴刻度
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        xticklabels=range(num_classes),
    #        yticklabels=range(num_classes),
    #        xlabel='Predicted Label',
    #        ylabel='True Label')
    # # 在每个格子中显示数值
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], 'd'),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    #
    # ax.set_title('Confusion Matrix')
    # plt.tight_layout()
    # # 设置保存路径
    # save_path = config.OUTPUT + '/' + valdata + '_confusion_map.png'
    # if save_path:
    #     plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
    # plt.close()
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

from tqdm import tqdm
from collections import defaultdict, Counter
@torch.no_grad()
def run_test_tda(loader, model, pos_shot_capacity=32, pos_alpha=2, pos_beta=5, val_index='shanxi_test', pos_cache=None):
    model.eval()
    # 初始化输出和目标列表
    all_outputs_org = []
    all_outputs = []
    all_targets = []
    data_results = []
    if pos_cache == None:
        pos_cache_index = True
    else:
        pos_cache_index = False
    with torch.no_grad():
        if pos_cache_index:
            pos_cache = defaultdict(lambda: defaultdict(list))

        # Test-time adaptation
        for i, (inp, mask, target, imgname) in enumerate(tqdm(loader, desc='Processed test images: ')):
            result = {}
            inp = (inp * mask).to(device)
            target = target.to(device)
            _clsout, image_features = model(inp)
            prob = torch.nn.functional.softmax(_clsout, dim=1).float()
            all_outputs_org.extend(prob.data.cpu().numpy())

            image_features /= image_features.norm(dim=-1, keepdim=True)
            pred = int(_clsout.topk(1, 1, True, True)[1].t()[0])
            loss = softmax_entropy(_clsout)
            pos = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']
            for posii in pos:
                if posii in imgname[0]:
                    name=posii
            if pos_cache_index:
                update_cache(pos_cache[name], pred, [image_features, loss], pos_shot_capacity)

            final_logits = _clsout.clone()

            if pos_cache[name]:
                final_logits += compute_cache_logits(image_features, pos_cache[name], _clsout.shape[-1], pos_alpha, pos_beta )

            prob = torch.nn.functional.softmax(final_logits, dim=1).float()
            all_outputs.extend(prob.data.cpu().numpy())
            # targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
            all_targets.extend(target.cpu().numpy())

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_outputs_org = np.array(all_outputs_org)
    all_targets = np.array(all_targets)
    # Path(config.OUTPUT + '/' + 'analyze').mkdir(parents=True, exist_ok=True)
    # df_data_results.to_csv(config.OUTPUT + '/' + val_index + '_dataprob_result.csv', index=False)
    accuracy, auc, per_acc, per_auc = adaptive_calculate_metrics(all_outputs_org, all_targets, val_index)
    print(f'{val_index} Original Accuracy: {accuracy:.4f}, AUC: {auc:.4f}', 'per_acc:', per_acc, 'per_auc:', per_auc)
    accuracy, auc, per_acc, per_auc = adaptive_calculate_metrics(all_outputs, all_targets, val_index)
    print(f'{val_index} After TDA Accuracy: {accuracy:.4f}, AUC: {auc:.4f}', 'per_acc:', per_acc, 'per_auc:', per_auc)
    return accuracy, auc, per_acc, per_auc
    # return loss_meter.avg, auc, acc, writer

if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    model_name = ['convnext']
    pt_names = ['ftda_model_best_acc_shanxi_train', 'ftda_model_best_acc_shanxi_test', 'ftda_model_best_acc_guizhou_test']

    args = parse_option()
    args.model_name = model_name[0]
    args.aug_rot = True
    args.aug_gaussian = True
    args.batch_size=60
    config = get_config(args)
    config.TRAIN.EPOCHS = 80
    config.OUTPUT = os.path.join(args.output, 'wmask_subcls_nodropout', args.model_name + '_databig_allsick_health',
                                 f'224_batch60_lr8e-05_testall')

    # pt_path='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/trainable_bestvalpt.pth'
    # pt_path ='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
    pt_path ='/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls_2global/224/contra_20251127/wmask_subcls_nodropout/convnext_databig_allsick_health/224_batch60_lr8e-05_testall/model_best_acc_guizhou_test.pth'
    # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-b/batch16_lr1e-05_testall/model_best_acc_shanxi_test.pth'
    # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/2models_distill/dinov3stu224_trained_convnext_slgmstea/224_batch10_lr1e-05_testall/model_best_acc_alltest.pth'
    # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout/3models_distill/dinov3st224_notrained_convnext_slgms_pneullmtea_0.5weight/224_batch10_lr3e-05_testall/model_best_acc_alltest.pth'
    # pt_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/qwen2_5vision_3B/best_224_batch16_lr1e-05_testall/model_best_acc_shanxi_val.pth'
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    for pt_name in pt_names:
        main(config,pt_names, pt_path)