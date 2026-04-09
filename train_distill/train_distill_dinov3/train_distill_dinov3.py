from train_distill.train_distill_dinov3.dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from train_distill.train_distill_dinov3.dinov2.models import build_model_from_cfg
from train_distill.train_distill_dinov3.dinov2.layers import DINOHead
from train_distill.train_distill_dinov3.dinov2.utils.utils import has_batchnorms
from train_distill.train_distill_dinov3.dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from train_distill.train_distill_dinov3.dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model, pretrained_fsdp_wrapper
from train_distill.train_distill_dinov3.dinov2.models.vision_transformer import BlockChunk
from train_distill.train_distill_dinov3.dinov2.fms import CONCH, UNI, Phikon
from train_distill.train_distill_dinov3.dinov2.kl_loss import get_kd_loss, FeatureLoss

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
    parser.add_argument('--batch-size', type=int, default=10, help="batch size for single GPU")
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
        transforms.RandomResizedCrop(512, scale=(0.75, 1.), interpolation=Image.BICUBIC),
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
    transform_list1=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    transform_list2 = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    global_transfo2 = transforms.Compose(transform_list)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset_train = Shanxi_wmask_Dataset2(args.data_root + 'seg_rec_img_1024', args.data_root + 'seg_rec_mask_1024',
                                       txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt',
                                       data_transform=global_transfo2, out_transform1=transform, out_transform2=transform_list2)

    dataset_train2 = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + 'seg_rec_mask_1024',
                                       txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt',
                                       data_transform=transform)

    dataset_val = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024',args.data_root + 'seg_rec_mask_1024',
                                 txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt', data_transform=transform)
    dataset_test = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024',args.data_root + 'seg_rec_mask_1024',
                                  txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt', data_transform=transform)
    dataset_test_dcm = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + 'seg_rec_mask_1024',
                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test_dcm.txt',
                                        data_transform=transform)

    dataset_test2 = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024','/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                   txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                   data_transform=transform)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
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
    data_loader_test_dcm = torch.utils.data.DataLoader(
        dataset_test_dcm,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    return data_loader_train, data_loader_train2, data_loader_val, data_loader_test, data_loader_test_dcm, data_loader_test2

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
from train_distill.train_distill_dinov3.dinov2.layers.dino_head import *
from model_cls.rad_dino.raddino import rad_dino_gh
from model_cls.convnext import *
from model_cls.vmambav2.vmamba import DINO_VSSM
# from model_cls.qwen2_5vl import qwen2_5vision
from torch import optim as optim
def main(config, ptname='model_ema_best_auc_shanxi_val'):
    data_loader_train, data_loader_train2, data_loader_val, data_loader_test, data_loader_test_dcm, data_loader_test2 = build_phe_loader(args)
    logger.info(f"Creating model:{args.model_name}")
    distiller_dict = {}
    distiller_dict['model_dinov3_student'] = vit_base(img_size=224,
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
    # distiller_dict['model_dinov3_student'].add_module('head', nn.Linear(768, 2))
    distiller_dict['model_convnext_teacher'] = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
    distiller_dict['model_slgms_teacher'] = DINO_VSSM()

    class MyModel(nn.Module):
        def __init__(self, teacher_model1, teacher_model2, student_model):
            super(MyModel, self).__init__()
            self.student = student_model
            self.teacher1 = teacher_model1
            self.teacher2 = teacher_model2

        def forward(self, x):
            # 在这里定义前向传播逻辑
            pass

    model = MyModel(distiller_dict['model_slgms_teacher'], distiller_dict['model_convnext_teacher'], distiller_dict['model_dinov3_student'])
    pretrained = '/disk3/wjr/workspace/sec_proj4/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    state_dict = torch.load(pretrained, map_location='cpu')
    msg = model.student.load_state_dict(state_dict, strict=False)
    print(msg)
    model.student.head = nn.Linear(768, 2)
    model.teacher2.head.fc = nn.Linear(768, 2)
    pretrained_student = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/selected_dcm_saomiao_900_health308_sick_592/224/contra_20250906/wmask_nosubcls_nodropout2/dinov3/vit-B/512_batch8_lr3e-05_testall/model_best_acc_guizhou_test.pth'
    state_dict = torch.load(pretrained_student, map_location='cpu', weights_only=False)['model']
    msg_student = model.student.load_state_dict(state_dict, strict=True)
    print(msg_student, pretrained_student)
    teacher_ckp_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
    checkpoint = torch.load(teacher_ckp_path, map_location="cpu", weights_only=False)
    msg_teacher = model.teacher2.load_state_dict(checkpoint['model'], strict=True)
    print(msg_teacher, teacher_ckp_path)

    teacher_ckp_path1 = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/checkpoint_best.pth.tar'
    state_dict = torch.load(teacher_ckp_path1, map_location="cpu", weights_only=False)['teacher']
    state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
    msg_teacher1 = model.teacher1.load_state_dict(state_dict, strict=False)
    print(msg_teacher1, teacher_ckp_path1)

    model.student.convnext_dinohead = DINOHead(768,768,nlayers=3,hidden_dim=2048,bottleneck_dim=256)
    model.student.slgms_dinohead = DINOHead(768, 768, nlayers=3, hidden_dim=2048, bottleneck_dim=256)

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
        model.student.load_state_dict(checkpoint['model'], strict=True)
        shanxi_train_accuracy, shanxi_train_sensitivity, shanxi_train_specificity, shanxi_train_auc, shanxi_train_f1, mcc_threshold = validate_final(
            config, data_loader_train2, model.student,
            threshold=0.5, valdata='shanxi_train')

        shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_val_auc, shanxi_val_f1, mcc_threshold = validate_final(
            config, data_loader_val, model.student,
            threshold=0.5, valdata='shanxi_val')
        shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc, shanxi_test_f1, shanxitest_mcc_threshold = validate_final(
            config, data_loader_test,
            model.student, threshold=0.5,
            valdata='shanxi_test')
        shanxi_test_accuracy_dcm, shanxi_test_sensitivity_dcm, shanxi_test_specificity_dcm, shanxi_test_auc_dcm, shanxi_test_f1_dcm, shanxitest_mcc_threshold_dcm = validate_final(
            config, data_loader_test_dcm,
            model.student, threshold=0.5,
            valdata='shanxi_test_dcm')
        guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity, guizhou_test_auc, guizhou_test_f1, shanxitest_mcc_threshold = validate_final(
            config, data_loader_test2,
            model.student, threshold=0.5,
            valdata='guizhou_test')
        return shanxi_train_auc, shanxi_train_accuracy, shanxi_train_sensitivity, shanxi_train_specificity, shanxi_val_auc, shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc_dcm, shanxi_test_accuracy_dcm, shanxi_test_sensitivity_dcm, shanxi_test_specificity_dcm, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity


    optimizer = build_optimizer(config, model.student)
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
    # for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    for epoch in range(config.TRAIN.START_EPOCH, 20):
        # data_loader_train.sampler.set_epoch(epoch)
        # train_one_epoch(config, model, rand_module, criterion, data_loader_train, optimizer, epoch, mixup_fn=None, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler, model_ema=model_ema)
        writer=train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn=None,
                        lr_scheduler=lr_scheduler, loss_scaler=loss_scaler,model_ema=model_ema,  writer=writer)
        # 保存当前RNG状态
        cpu_state, cuda_state = save_rng_states()
        # loss, auc, acc, writer = validate(data_loader_train2, model, epoch, writer, index='shanxi_train')
        # loss, acc_ema, auc_ema, writer = validate(
        #     data_loader_train2, model_ema.ema, epoch, writer,
        #     index='shanxi_train_ema')
        loss, auc, acc, writer = validate(data_loader_val, model.student,epoch, writer, index='shanxi_val')
        loss, auc_ema, acc_ema, writer = validate(data_loader_val, model_ema.ema.student, epoch, writer, index='shanxi_val_ema')
        # loss, auc, acc, writer = validate(data_loader_test, model, epoch, writer, index='shanxi_test')
        # loss, auc_ema, acc_ema, writer = validate(data_loader_test, model_ema.ema, epoch, writer, index='shanxi_test_ema')
        test_loss, test_auc,test_acc, writer = validate(data_loader_test, model.student, epoch, writer, index='shanxi_test')
        test_loss, test_auc_ema, test_acc_ema, writer = validate(data_loader_test, model_ema.ema.student, epoch, writer, index='shanxi_test_ema')
        loss_test2, auc_test2, test_acc2,writer = validate(data_loader_test2, model.student, epoch, writer, index='guizhou_test')
        loss_test2, auc_test2_ema, test_acc2_ema, writer = validate(data_loader_test2, model_ema.ema.student, epoch, writer,
                                                            index='guizhou_test_ema')
        avg_test_acc = (test_acc * 3847 + test_acc2 * 3144) / 6991
        avg_test_acc_ema = (test_acc_ema * 3847 + test_acc2_ema * 3144) / 6991
        writer.add_scalar('all_test acc_threshold0.5', avg_test_acc, epoch)
        writer.add_scalar('all_test_ema acc_threshold0.5', avg_test_acc_ema, epoch)
        val_avg = acc
        logger.info(f"Task average of the network on the val images: {val_avg:.4f}")
        if acc > max_acc_avg:
            best_epoch = epoch
            max_acc_avg = max(max_acc_avg, acc)
            save_test_checkpoint(config, epoch, model.student, max_acc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_shanxi_val.pth')
        else:
            max_acc_avg = max(max_acc_avg, acc)

        if avg_test_acc > max_acc_ema_avg2:
            best_epoch = epoch
            max_acc_ema_avg2 = max(max_acc_ema_avg2, avg_test_acc)
            save_test_checkpoint(config, epoch, model.student, max_acc_ema_avg2, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_alltest.pth')
        else:
            max_acc_ema_avg2 = max(max_acc_ema_avg2, avg_test_acc)

        if test_acc > max_acc_ema_avg:
            best_epoch = epoch
            max_acc_ema_avg = max(max_acc_ema_avg, test_acc)
            save_test_checkpoint(config, epoch, model.student, max_acc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_shanxi_test.pth')
        else:
            max_acc_ema_avg = max(max_acc_ema_avg, test_acc)

        if avg_test_acc_ema > max_auc_ema_avg3:
            best_epoch = epoch
            max_auc_ema_avg3 = max(max_auc_ema_avg3, avg_test_acc_ema)
            save_test_checkpoint(config, epoch, model_ema.ema.student, max_auc_ema_avg3, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_ema_best_acc_alltest.pth')
        else:
            max_auc_ema_avg3 = max(max_auc_ema_avg3, avg_test_acc_ema)
        #
        # if test_acc2_ema > max_auc_ema_avg4:
        #     best_epoch = epoch
        #     max_auc_ema_avg4 = max(max_auc_ema_avg4, test_acc2_ema)
        #     save_test_checkpoint(config, epoch, model_ema.ema, max_auc_ema_avg4, optimizer, lr_scheduler,
        #                          loss_scaler,
        #                          logger, ptname='model_ema_best_acc_guizhou_test.pth')
        # else:
        #     max_auc_ema_avg4 = max(max_auc_ema_avg4, test_acc2_ema)

        if auc > max_auc_avg:
            best_epoch = epoch
            max_auc_avg = max(max_auc_avg, acc)
            save_test_checkpoint(config, epoch, model.student, max_auc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_auc_shanxi_val.pth')
        else:
            max_auc_avg = max(max_auc_avg, auc)

        if test_acc2 > max_auc_ema_avg:
            best_epoch = epoch
            max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
            save_test_checkpoint(config, epoch, model.student, max_auc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_guizhou_test.pth')
        else:
            max_auc_ema_avg = max(max_auc_ema_avg, test_acc2)
        logger.info(f'Best epoch: {best_epoch}'+' Max acc average: '+"%.4f" % max_acc_avg+ ' Test auc average: '+"%.4f" % max_auc_avg)
        restore_rng_states(cpu_state, cuda_state)
    ckp_path = os.path.join(config.OUTPUT, ptname+'.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu",weights_only=False)
        model.student.load_state_dict(checkpoint['model'], strict=True)
    shanxi_train_accuracy, shanxi_train_sensitivity, shanxi_train_specificity, shanxi_train_auc, shanxi_train_f1, mcc_threshold = validate_final(
        config, data_loader_train2, model.student,
        threshold=0.5, valdata='shanxi_train')

    shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_val_auc, shanxi_val_f1, mcc_threshold = validate_final(
        config, data_loader_val, model.student,
        threshold=0.5, valdata='shanxi_val')
    shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc, shanxi_test_f1, shanxitest_mcc_threshold = validate_final(
        config, data_loader_test,
        model.student, threshold=0.5,
        valdata='shanxi_test')
    shanxi_test_accuracy_dcm, shanxi_test_sensitivity_dcm, shanxi_test_specificity_dcm, shanxi_test_auc_dcm, shanxi_test_f1_dcm, shanxitest_mcc_threshold_dcm = validate_final(
        config, data_loader_test_dcm,
        model.student, threshold=0.5,
        valdata='shanxi_test_dcm')
    guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity, guizhou_test_auc, guizhou_test_f1, shanxitest_mcc_threshold = validate_final(
        config, data_loader_test2,
        model.student, threshold=0.5,
        valdata='guizhou_test')
    return shanxi_train_auc, shanxi_train_accuracy, shanxi_train_sensitivity, shanxi_train_specificity, shanxi_val_auc, shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc_dcm, shanxi_test_accuracy_dcm, shanxi_test_sensitivity_dcm, shanxi_test_specificity_dcm, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity

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

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, writer=None):
    model.student.train()
    model.teacher2.eval()
    model.teacher1.eval()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    criterion2 = OrdinalMetricLoss()
    start = time.time()
    koleo_loss = KoLeoLoss()
    end = time.time()
    for idx, (images, masks,images2, masks2, targets, _) in enumerate(data_loader):
        # rand_module.randomize()
        samples = (images * masks).to(device)
        samples2 = (images2 * masks2).to(device)
        # images = images.to(device)
        targets = targets[:,1].long().to(device)
        data_time.update(time.time() - end)

        # outputs= model.student(samples)
        cls_token, patch_token, outputs = model.student(samples, is_training=True)
        loss_koleo=koleo_loss(cls_token)
        with torch.no_grad():
            outputs_teacher,_,_,cls_token_teacher, patch_token_teacher = model.teacher2(samples2)
            outputs_teacher1, patch_token_teacher1, cls_token_teacher1= model.teacher1(samples2, distill_token=True)

        loss_cls = criterion(outputs, targets)
        log_pred_student = F.log_softmax(outputs, dim=1)
        pred_teacher = F.softmax(outputs_teacher, dim=1)
        pred_teacher1 = F.softmax(outputs_teacher1, dim=1)
        loss_kd1 = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd2 = F.kl_div(log_pred_student, pred_teacher1, reduction="none").sum(1).mean()
        loss = loss_cls+0.5*(loss_kd1+loss_kd2)+0.1*koleo_loss
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
        loss_cls_meter.update(loss_cls.item(), targets.size(0))
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
                f'loss_cls {loss_cls_meter.val:.4f} ({loss_cls_meter.avg:.4f})\t'
                # f'loss_map_meter {loss_map_meter.val:.4f} ({loss_map_meter.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                )
            writer.add_scalar('Shanxitrain cls loss', loss_cls.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain convnexttea_kd loss', loss_kd1.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain slgmstea_kd loss', loss_kd2.item(), epoch * num_steps + idx)
            # writer.add_scalar('Shanxitrain cam_contra_sick_health_loss', loss_contraii.item(), epoch * num_steps + idx)


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
def validate(data_loader, model,epoch, writer, index='shanxi_val'):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, _) in enumerate(data_loader):
        # rand_module.randomize()
        images = (images * masks).to(device)
        # images = images.to(device)
        labels = targets.to(device)
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred = model(images)
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
    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=index,
        plot_roc=True,
        save_fpr_tpr=True)
    writer.add_scalar(index + ' auc', auc, epoch)
    writer.add_scalar(index + ' acc_threshold0.5', acc, epoch)
    writer.add_scalar(index + ' f1', f1, epoch)
    writer.add_scalar(index + ' sensitivity_threshold0.5', sensitivity, epoch)
    writer.add_scalar(index + ' specificity_threshold0.5', specificity, epoch)
    # writer.add_scalar(index + ' threshold', threshold, epoch)
    logger.info(
        f'{index} epoch: {epoch}' + ' auc: ' + "%.4f" % auc + ' acc_threshold0.5: ' + "%.4f" % acc)

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
def adaptive_calculate_metrics(all_outputs, all_targets_onehot,threshold=None, valdata='shanxi_test',num_classes=2, plot_roc=True, save_fpr_tpr=True):
    # # 将输出通过 Sigmoid 激活函数转换为概率
    # # probabilities = torch.sigmoid(all_outputs).cpu().numpy()
    # probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    # # probabilities = all_outputs.cpu().numpy()
    # all_targets = all_targets_onehot.cpu().numpy()
    probabilities=all_outputs
    all_targets=all_targets_onehot
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
            plt.scatter(specific_fpr, specific_tpr, marker='*', color='red', s=200)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            plt.legend(loc='lower right', prop={'size': 20})
            save_path = config.OUTPUT + '/' + valdata + '_roc_map.pdf'
            plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
            # # 显示
            # plt.show()
            # # 自动关闭图像
            # plt.close()

        auc_scores.append(auc)
        # # 找到使得 Sensitivity 和 Specificity 最接近的阈值
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        # print(optimal_threshold)
        # optimal_thresholds.append(optimal_threshold)
        # if args.valdata == 'shanxi':
        #     optimal_threshold=0.5
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
        np.save(config.OUTPUT + '/' + valdata + '_fpr.npy', fpr)
        np.save(config.OUTPUT + '/' + valdata + '_tpr.npy', tpr)
    # # 打印每个类别的指标
    # if CLASS_NAMES is not None:
    #     for i in range(num_classes):
    #         print(f'Class {CLASS_NAMES[i]}:')
    #         print(f'  AUROC: {auc_scores[i]:.4f}')
    #         print(f'  Sensitivity: {sensitivity_scores[i]:.4f}')
    #         print(f'  Specificity: {specificity_scores[i]:.4f}')
    #         print(f'  Accuracy: {acc_scores[i]:.4f}')
    #         print(f'  F1 Score: {f1_scores[i]:.4f}')
    #         print(f'  Precision: {precision_scores[i]:.4f}')
    #         print(f'  Recall: {recall_scores[i]:.4f}')
    #         print(f'  Optimal Threshold: {optimal_thresholds[i]:.4f}')
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
def validate_final(config, data_loader, model,threshold=None, valdata='shanxi_test'):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, _) in enumerate(data_loader):
        # rand_module.randomize()
        images = (images * masks).to(device)
        labels = targets.to(device)
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred = model(images)
        if args.model_name == "dinov2":
            pred = pred[1]
        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.float().data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # all_outputs.append(pred)
        # all_targets.append(labels)
        # loss = nn.CrossEntropyLoss()(pred, labels)
        loss = nn.BCEWithLogitsLoss()(pred, labels)

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
    accuracy, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(all_outputs,
                                                                                                all_targets,
                                                                                                threshold=threshold,
                                                                                                valdata=valdata,
                                                                                                plot_roc=True,
                                                                                                save_fpr_tpr=True)

    avg = (accuracy + auc + sensitivity + specificity) / 4
    # 打印结果
    print(f'AVG: {avg:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
          f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, '
          f'F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return accuracy, sensitivity, specificity, auc, f1, threshold_mcc

if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    # lrs = [5e-5]
    lrs = [1e-5,3e-5, 5e-5,8e-5, 1e-4,  3e-4]
    batch_sizes = [10]
    # model_name = ['hrnet18','resnet50', 'vit-base','swintiny']
    model_name = ['dinov3',]
    aug_rot=[True]
    aug_gaussian=[True]
    loss_type=['ce']
    pt_names=['model_best_acc_alltest','model_ema_best_acc_alltest','model_best_acc_shanxi_val','model_best_auc_shanxi_val','model_best_acc_shanxi_test','model_best_acc_guizhou_test']
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
                    config.OUTPUT = os.path.join(args.output, 'wmask_nosubcls_nodropout2','2models_distill','dinov3stu512_trained_convnext_slgmstea',
                                                 '512_batch10_lr' + str(args.base_lr)+'_testall')

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
                        shanxi_train_auc, shanxi_train_acc, shanxi_train_sens, shanxi_train_spec, shanxi_val_auc, shanxi_val_acc, shanxi_val_sens, shanxi_val_spec, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity,  shanxi_test_auc_dcm, shanxi_test_accuracy_dcm, shanxi_test_sensitivity_dcm, shanxi_test_specificity_dcm,guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity = main(
                            config, pt_name)
                        shanxi_train_avgs = (
                                                    shanxi_train_auc + shanxi_train_acc + shanxi_train_sens + shanxi_train_spec) / 4

                        shanxi_val_avgs = (shanxi_val_auc + shanxi_val_acc + shanxi_val_sens + shanxi_val_spec) / 4

                        shanxi_avgs = (
                                              shanxi_test_auc + shanxi_test_accuracy + shanxi_test_sensitivity + shanxi_test_specificity) / 4

                        shanxi_dcm_avgs = (
                                              shanxi_test_auc_dcm + shanxi_test_accuracy_dcm + shanxi_test_sensitivity_dcm + shanxi_test_specificity_dcm) / 4

                        guizhou_avgs = (
                                               guizhou_test_auc + guizhou_test_accuracy + guizhou_test_sensitivity + guizhou_test_specificity) / 4
                        avg_test_acc = (shanxi_test_accuracy * 3847 + guizhou_test_accuracy * 3144) / 6991
                        print('avg_test_acc:', avg_test_acc)

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
                            re = f'{args.model_name} {args.base_lr} {args.aug_rot}_augrot {args.aug_gaussian}_aug_gaussian: threshold: 0.5, {pt_name}:\n' \
                                 f'shanxi_train_avgs: {shanxi_train_avgs}, shanxi_train_auc: {shanxi_train_auc}, shanxi_train_acc: {shanxi_train_acc}, shanxi_train_sensitivity: {shanxi_train_sens}, shanxi_train_specificity: {shanxi_train_spec}\n' \
                                 f'shanxi_val_avgs: {shanxi_val_avgs}, shanxi_val_auc: {shanxi_val_auc}, shanxi_val_acc: {shanxi_val_acc}, shanxi_val_sensitivity: {shanxi_val_sens}, shanxi_val_specificity: {shanxi_val_spec}\n' \
                                 f'shanxi_guizhou_allacc: {avg_test_acc}\n' \
                                 f'shanxi_test_avg: {shanxi_avgs}, shanxi_test_auc: {shanxi_test_auc}, shanxi_test_acc: {shanxi_test_accuracy}, shanxi_test_sensitivity: {shanxi_test_sensitivity}, shanxi_test_specificity: {shanxi_test_specificity}\n' \
                                 f'shanxi_dcm_test_avg: {shanxi_dcm_avgs}, shanxi_dcm_test_auc: {shanxi_test_auc_dcm}, shanxi_dcm_test_acc: {shanxi_test_accuracy_dcm}, shanxi_dcm_test_sensitivity: {shanxi_test_sensitivity_dcm}, shanxi_dcm_test_specificity: {shanxi_test_specificity_dcm}\n' \
                                 f'guizhou_test_avg: {guizhou_avgs}, guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_sensitivity: {guizhou_test_sensitivity}, guizhou_test_specificity: {guizhou_test_specificity}\n\n'
                            file.write(re)




