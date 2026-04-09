'''shot: Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation'''
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
sys.path.append('/disk3/wjr/workspace/sec_nejm/nejm_baseline_wjr/')
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
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
import os
from train_resnet.gradcam.gradcam import GradCAM, GradCAM_twos
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

def reshape_transform(tensor):
    # 去掉cls token
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    # tensor=torch.cat((tensor[:, :98, :],tensor[:, 99:, :]),dim=1)
    # # tensor = tensor[:, 1:, :]
    hw,c=tensor.shape
    h=int(np.sqrt(hw))

    result = tensor.reshape(
    h, h, tensor.size(-1))

    # 将通道维度放到第一个位置
    result = result.permute(2,0,1)
    return result
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--model_name', type=str,
                        default='resnet50',
                        help='t')
    # easy config modification
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra/subregions_mixed/model_ema_999_wcamloss_5class_1classifier_cloc/', type=str, metavar='PATH',
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
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    args, unparsed = parser.parse_known_args()
    return args
from torchvision import transforms as pth_transforms
from PIL import ImageFilter, ImageOps
from PIL import Image
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
    global_transfo2 = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
        GaussianBlur(0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    global_transfo2_subregions = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
        GaussianBlur(0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    global_transfo_subregions = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # dataset_train = Shanxi_w7masks_Subregions_MixedSickHealth_Dataset(args.data_root + 'seg_rec_img_1024',
    #                                                   args.data_root + 'seg_rec_mask_1024',
    #                                txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt',
    #                                csvpath=args.data_root + 'subregions_label_shanxi_new.xlsx',
    #                                data_transform=global_transfo2, data_subregions_transform=global_transfo2_subregions,
    #                                sub_img_size=128,)
    dataset_train = Shanxi_w7masks_Subregions_5classes_Mixednew_Dataset(args.data_root + 'seg_rec_img_1024',
                                                      args.data_root + 'seg_rec_mask_1024',
                                   txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt',
                                   csvpath=args.data_root + 'subregions_label_shanxi_new.xlsx',
                                   data_transform=global_transfo2, data_subregions_transform=global_transfo2_subregions,
                                   sub_img_size=128,)
    dataset_train2 = Shanxi_w7masks_Subregions_5classes_Dataset(args.data_root + 'seg_rec_img_1024',
                                                                        args.data_root + 'seg_rec_mask_1024',
                                                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt',
                                                                        csvpath=args.data_root + 'subregions_label_shanxi_new.xlsx',
                                                                        data_transform=global_transfo2,
                                                                        data_subregions_transform=global_transfo2_subregions,
                                                                        sub_img_size=128, )
    dataset_val = Shanxi_w7masks_Subregions_5classes_Dataset(args.data_root + 'seg_rec_img_1024',
                                       args.data_root + 'seg_rec_mask_1024',
                                       txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
                                       csvpath=args.data_root + 'subregions_label_shanxi_new.xlsx',
                                       data_transform=transform, data_subregions_transform=global_transfo_subregions,
                                       sub_img_size=128,)
    dataset_val2 = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024',
                                        args.data_root + 'seg_rec_mask_1024',
                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
                                        data_transform=transform)
    dataset_test = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024',
                                        args.data_root + 'seg_rec_mask_1024',
                                        txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/test.txt',
                                        data_transform=transform)
    dataset_test3 = Shanxi_w7masks_Subregions_5classes_Dataset(
        '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
        '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
        txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
        csvpath='/disk3/wjr/dataset/nejm/guizhoudataset/subregions_guizhou.xlsx',
        data_transform=transform,
        data_subregions_transform=global_transfo_subregions,
        sub_img_size=128, )
    dataset_test2 = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                         '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                         txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                         data_transform=transform)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)
    sampler_train2 = torch.utils.data.DistributedSampler(dataset_train2, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=True)
    sampler_val2 = torch.utils.data.DistributedSampler(dataset_val2, num_replicas=num_tasks, rank=global_rank,
                                                       shuffle=True)

    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank,
                                                       shuffle=True)
    sampler_test2 = torch.utils.data.DistributedSampler(dataset_test2, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)
    sampler_test3 = torch.utils.data.DistributedSampler(dataset_test3, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2, sampler=sampler_train,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val2 = torch.utils.data.DataLoader(
        dataset_val2, sampler=sampler_val2,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2, sampler=sampler_test2,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test3 = torch.utils.data.DataLoader(
        dataset_test3, sampler=sampler_test3,
        batch_size=30,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_train, data_loader_train2, data_loader_val, data_loader_val2, data_loader_test, data_loader_test2, data_loader_test3


from model_cls.resnet50 import resnet50
from model_cls.convnext import *
from model_cls.swin import *
from model_cls.swinv2 import *
from timm.utils import ModelEma as ModelEma
from utils.cloc import *
from scipy.spatial.distance import cdist
def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def main(config):
    data_loader_train,data_loader_train2,  data_loader_val, data_loader_val2, data_loader_test, data_loader_test2, data_loader_test3 = build_phe_loader(
        args)
    logger.info(f"Creating model:{args.model_name}")
    if args.model_name=='resnet50':
        model = resnet50(pretrained=True)
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == 'hrnet18':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w18_3rdparty_8xb32-ssld_in1k_20220120-455f69ea.pth'
        model = get_model('hrnet-w18_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == 'hrnet48':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth'
        model = get_model('hrnet-w48_3rdparty_8xb32-ssld_in1k', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == "convnext":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model1 = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        model = convnexttiny(pretrained=True)
        model.head.fc = nn.Linear(768, 5)
        learnable_map = [
            ['fixed', 0.4],
            ['fixed', 0.4],
            ['fixed', 0.4],
            ['fixed', 0.4],
        ]
        model.margin_criterion = OrdinalContrastiveLoss_mm(
            n_classes=5,
            device=device,
            # learnable_map=learnable_map
        )
        # model.sub_cls = nn.Linear(768, 5)
    elif args.model_name == "convnext_huge":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        model = get_model('convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-512px', pretrained=pretrained_cfg)
        model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "dinov2":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'
        model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=pretrained_cfg)
        model.backbone.ln1 = nn.Linear(384, 2, bias=True)
    elif args.model_name == "swintiny":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        # model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 2, bias=True)
        model = swintiny(pretrained=True)
        model.head.fc = nn.Linear(768, 2, bias=True)
    elif args.model_name == "swintinyv2":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        # model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 2, bias=True)
        model = swintinyv2(pretrained=True)
        model.head.fc = nn.Linear(768, 2, bias=True)
    elif args.model_name == "vit-base":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth'
        model = get_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', pretrained=pretrained_cfg)
        model.head.layers.head = nn.Linear(768, 2, bias=True)
    model.to(device)
    model_ema = None
    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        device='cpu' if args.model_ema_force_cpu else '',
        resume='')
    print("Using EMA with decay = %.8f" % args.model_ema_decay)
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    ckp_path = os.path.join(config.OUTPUT, f'model_ema_best_acc_shanxi_val.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        msg=model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
        optimizer = build_optimizer(config, model)
        loss_scaler = NativeScalerWithGradNormCount()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        for k, v in model.named_parameters():
            if k.__contains__("head.fc"):
                v.requires_grad = True
            else:
                v.requires_grad = False
        # criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCEWithLogitsLoss()
        test_avg = 0
        logger.info("Start training")
        _ = validate_val_shot(data_loader_test, model, 0, writer,
                              val_index='shanxitest')
        _ = validate_val_shot(data_loader_test2, model, 0, writer,
                              val_index='guizhoutest')
        for epoch in range(config.TRAIN.START_EPOCH, 5):
            data_loader_train.sampler.set_epoch(epoch)
            # _ = validate_val_shot(data_loader_val2, model, 0, writer,
            #                       val_index='shanxitest')
            # train_one_epoch(config, model, rand_module, criterion, data_loader_train, optimizer, epoch, mixup_fn=None, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler, model_ema=model_ema)
            train_one_epoch(config, model, criterion, data_loader_train2, optimizer, epoch, mixup_fn=None,
                            lr_scheduler=lr_scheduler, loss_scaler=loss_scaler)
            _ = validate_val_shot(data_loader_test, model, 0, writer,
                                  val_index='shanxitest')
            _ = validate_val_shot(data_loader_test2, model, 0, writer,
                                  val_index='guizhoutest')


import diffsort
import torch.nn.functional as F
from collections import defaultdict
# 假设 all_sub_fea 是这样的列表：[{'0': tensor}, {'1': tensor}, ...]
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
def obtain_label(loader, model):
    model.eval()
    start_test = True
    with torch.no_grad():
        for idx, (images,masks,  targets, _) in enumerate(loader):
            images = (images*masks).to(device)
            labels = targets[:,1]
            outputs, feas = model(images)
            feas = model.neck(feas).flatten(start_dim=1, end_dim=-1)
            sick_outputs = outputs[:, :3].sum(dim=-1)
            health_outputs = outputs[:, 3:].sum(dim=-1)
            pred = torch.stack([sick_outputs, health_outputs], dim=1)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = pred.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, pred.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    all_output = nn.Softmax(dim=-1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(2)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>30)
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

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    model.train()
    return pred_label.astype('int') #, labelset
def obtain_sub_label(loader, model):
    model.eval()
    start_test = True
    start_sub_test = True
    with torch.no_grad():
        for idx, (images,masks,targetss, subregions_labelss, masks_index,_) in enumerate(loader):
            subimgs = torch.cat(images[1:], dim=0).to(device)
            submasks = torch.cat(masks[1:], dim=0).to(device)
            subregions_labels = torch.cat(subregions_labelss, dim=0).squeeze(-1).to(device)
            subimgs = subimgs[subregions_labels != -1]
            submasks = submasks[subregions_labels != -1]
            subregions_labels = subregions_labels[subregions_labels != -1]
            if subregions_labels.any():
                sub_outputs, subregions_fea = model(subimgs * submasks)
                sub_feas = model.neck(subregions_fea).flatten(start_dim=1, end_dim=-1)
                if start_sub_test:
                    all_sub_fea = [{f'{idx}': sub_feas.float().cpu()}]
                    all_sub_output =[{f'{idx}': sub_outputs.float().cpu()}]
                    all_sub_label = [{f'{idx}': subregions_labels.float().cpu()}]
                    start_sub_test = False
                else:
                    all_sub_fea.append({f'{idx}': sub_feas.float().cpu()})
                    all_sub_output.append({f'{idx}': sub_outputs.float().cpu()})
                    all_sub_label.append({f'{idx}': subregions_labels.float().cpu()})

            images = images[0].to(device)
            for ii in range(len(targetss)):
                targetss[ii] = targetss[ii].to(device)
            targets = targetss[0].to(device)
            for ii in range(len(subregions_labelss)):
                subregions_labelss[ii] = subregions_labelss[ii].to(device)
            if isinstance(masks[0], list):
                masks = masks[0]
                for ii in range(len(masks)):
                    masks[ii] = masks[ii].to(device)
            else:
                masks = masks[0].to(device)
            images = (images*masks[0]).to(device)
            labels = targets[:,1].to(device)
            outputs, feas = model(images)
            feas = model.neck(feas).flatten(start_dim=1, end_dim=-1)
            sick_outputs = outputs[:, :3].sum(dim=-1)
            health_outputs = outputs[:, 3:].sum(dim=-1)
            pred = torch.stack([sick_outputs, health_outputs], dim=1)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = pred.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, pred.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)



    # 分别合并 fea、output、label
    final_sub_fea = merge_dict_list(all_sub_fea)
    final_sub_output = merge_dict_list(all_sub_output)
    final_sub_label = merge_dict_list(all_sub_label)

    _, subpredict = torch.max(final_sub_output, 1)
    final_sub_fea = (final_sub_fea.t() / torch.norm(final_sub_fea, p=2, dim=1)).t()
    sub_accuracy = torch.sum(torch.squeeze(subpredict).float() == final_sub_label).item() / float(final_sub_label.size()[0])

    final_sub_fea = final_sub_fea.float().cpu().numpy()
    K = final_sub_output.size(1)
    aff = final_sub_output.float().cpu().numpy()
    initc = aff.transpose().dot(final_sub_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    subcls_count = np.eye(K)[subpredict].sum(axis=0)
    labelset = np.where(subcls_count > 0)
    labelset = labelset[0]
    # print(labelset)
    dd = cdist(final_sub_fea, initc[labelset], 'cosine')
    sub_pred_label = dd.argmin(axis=1)
    sub_pred_label = labelset[sub_pred_label]

    for round in range(1):
        aff = np.eye(K)[sub_pred_label]
        initc = aff.transpose().dot(final_sub_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(final_sub_fea, initc[labelset], 'cosine')
        sub_pred_label = dd.argmin(axis=1)
        sub_pred_label = labelset[sub_pred_label]

    sub_acc = np.sum(sub_pred_label == final_sub_label.float().numpy()) / len(all_fea)
    sub_predict_dict=merge_list2dict(sub_pred_label, all_sub_label)


    all_output = nn.Softmax(dim=-1)(all_output)
    # ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    # unknown_weight = 1 - ent / np.log(2)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>30)
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

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    log_str = 'SUB Accuracy = {:.2f}% -> {:.2f}%'.format(sub_accuracy * 100, sub_acc * 100)
    print(log_str+'\n')
    model.train()
    return pred_label.astype('int'), sub_predict_dict #, labelset
from mmpretrain import FeatureExtractor
def train_one_epoch_nosub(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    loss_entropy_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()
    end = time.time()
    model.eval()
    mem_label = obtain_label(data_loader, model)
    mem_label = torch.from_numpy(mem_label).to(device)
    model.train()
    for idx, (images, masks,targets, _) in enumerate(data_loader):
        images = (images*masks).to(device)
        targets = targets.to(device)
        outputs_test, feas = model(images)
        feas = model.neck(feas).flatten(start_dim=1, end_dim=-1)
        sick_outputs = outputs_test[:, :3].sum(dim=-1)
        health_outputs = outputs_test[:, 3:].sum(dim=-1)
        outputs_test = torch.stack([sick_outputs, health_outputs], dim=1)
        pred = mem_label[idx*30:idx*30+images.shape[0]]
        classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        entropy_loss = torch.mean(Entropy(softmax_out))
        loss=classifier_loss* 0.3 + entropy_loss * 1.
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        loss_cls_meter.update(classifier_loss.item(), targets.size(0))
        loss_entropy_meter.update(entropy_loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
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
                f'loss_entropy_meter {loss_entropy_meter.val:.4f} ({loss_entropy_meter.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    loss_entropy_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()
    end = time.time()
    model.eval()
    mem_label, mem_sub_label = obtain_sub_label(data_loader, model)
    mem_label = torch.from_numpy(mem_label).to(device)
    # mem_sub_label = torch.from_numpy(mem_sub_label).to(device)

    model.train()
    for idx, (images,masks, targetss, subregions_labelss, masks_index,_) in enumerate(data_loader):
        subimgs = torch.cat(images[1:], dim=0).to(device)
        submasks = torch.cat(masks[1:], dim=0).to(device)
        subregions_labels = torch.cat(subregions_labelss, dim=0).squeeze(-1).to(device)
        subimgs = subimgs[subregions_labels != -1]
        submasks = submasks[subregions_labels != -1]
        subregions_labels = subregions_labels[subregions_labels != -1]
        sub_loss=0
        if subregions_labels.any():
            sub_outputs, subregions_fea = model(subimgs * submasks)
            sub_feas = model.neck(subregions_fea).flatten(start_dim=1, end_dim=-1)
            _,sub_pred_v = list(mem_sub_label[idx][0].items())[0]
            sub_pred_v=torch.tensor(sub_pred_v).to(device)
            # aa=pred.items()
            sub_classifier_loss = nn.CrossEntropyLoss()(sub_outputs, sub_pred_v)
            sub_softmax_out = nn.Softmax(dim=1)(sub_outputs)
            sub_entropy_loss = torch.mean(Entropy(sub_softmax_out))
            sub_loss = sub_classifier_loss * 0.3 +sub_entropy_loss * 1.
            sub_loss = sub_loss / config.TRAIN.ACCUMULATION_STEPS

        images = images[0].to(device)
        for ii in range(len(targetss)):
            targetss[ii] = targetss[ii].to(device)
        targets = targetss[0].to(device)
        for ii in range(len(subregions_labelss)):
            subregions_labelss[ii] = subregions_labelss[ii].to(device)
        if isinstance(masks[0], list):
            masks = masks[0]
            for ii in range(len(masks)):
                masks[ii] = masks[ii].to(device)
        else:
            masks = masks[0].to(device)
        images = (images * masks[0]).to(device)
        outputs_test, feas = model(images)
        # feas = model.neck(feas).flatten(start_dim=1, end_dim=-1)
        sick_outputs = outputs_test[:, :3].sum(dim=-1)
        health_outputs = outputs_test[:, 3:].sum(dim=-1)
        outputs_test = torch.stack([sick_outputs, health_outputs], dim=1)
        pred = mem_label[idx*30:idx*30+images.shape[0]]
        classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        entropy_loss = torch.mean(Entropy(softmax_out))
        loss=classifier_loss* 0.3 + entropy_loss * 1.
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        loss_cls_meter.update(classifier_loss.item(), targets.size(0))
        loss_entropy_meter.update(entropy_loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
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
                f'loss_entropy_meter {loss_entropy_meter.val:.4f} ({loss_entropy_meter.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score
import numpy as np
import torch

def calculate_auc_metrics(all_outputs, all_targets_onehot):
    probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    all_targets = all_targets_onehot.cpu().numpy()
    auc = roc_auc_score(all_targets[:, 0], probabilities[:, 0])
    return auc * 100
@torch.no_grad()
def validate_val_shot(data_loader, model,epoch, writer, val_index='shanxi_val'):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    start_test = True
    for idx, (images,masks, targets,_) in enumerate(data_loader):
        images = (images*masks).to(device)
        labels = targets.to(device)
        outputs, feas = model(images)
        feas = model.neck(feas).flatten(start_dim=1, end_dim=-1)
        sick_outputs = outputs[:, :3].sum(dim=-1)
        health_outputs = outputs[:, 3:].sum(dim=-1)
        pred = torch.stack([sick_outputs, health_outputs], dim=1)
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
    unknown_weight = 1 - ent / np.log(2)
    _, predict = torch.max(all_output, 1)
    # predict[predict==1]=0
    # predict[predict == 2] = 0
    # predict[predict == 3] = 1
    # predict[predict == 4] = 1
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
    labelset = np.where(cls_count>30)
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

@torch.no_grad()
def validate(data_loader, model,epoch, writer, val_index='shanxi_val'):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images,masks, targets,_) in enumerate(data_loader):
        images = (images*masks).to(device)
        labels = targets.to(device)
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred,_ = model(images)
        sick_outputs = pred[:, :3].sum(dim=-1)
        health_outputs = pred[:, 3:].sum(dim=-1)
        pred = torch.stack([sick_outputs, health_outputs], dim=1)
        if args.model_name == "dinov2":
            pred = pred[1]
        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # # 收集输出和目标
        # all_outputs.append(pred)
        # all_targets.append(labels)
        loss = nn.CrossEntropyLoss()(pred, labels)

        loss_meter.update(loss.item(), labels.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    # 计算整体指标
    # 计算整体指标
    # 计算整体指标
    auc = roc_auc_score(all_targets[:, 0], all_outputs[:, 0])
    fpr, tpr, thresholds = roc_curve(all_targets[:, 0], all_outputs[:, 0])
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (all_outputs[:, 0] >= 0.5).astype(int)
    # y_pred = (probabilities[:, i] >= 0.5).astype(int)
    # 计算各项指标
    acc = accuracy_score(all_targets[:, 0], y_pred)
    true_positive = ((y_pred == 1) & (all_targets[:, 0] == 1)).sum()
    true_negative = ((y_pred != 1) & (all_targets[:, 0] != 1)).sum()
    false_positive = ((y_pred == 1) & (all_targets[:, 0] != 1)).sum()
    false_negative = ((y_pred != 1) & (all_targets[:, 0] == 1)).sum()

    sensitivity=(
        true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0)
    specificity=(
        true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0)

    writer.add_scalar(val_index + ' auc', auc, epoch)
    writer.add_scalar(val_index + ' acc_threshold0.5', acc, epoch)
    writer.add_scalar(val_index + ' sensitivity_threshold0.5', sensitivity, epoch)
    writer.add_scalar(val_index + ' specificity_threshold0.5', specificity, epoch)
    writer.add_scalar(val_index + ' threshold', threshold, epoch)
    logger.info(
        f'{val_index} epoch: {epoch}' + ' auc: ' + "%.4f" % auc + ' acc_threshold0.5: ' + "%.4f" % acc)

    # auc = calculate_auc_metrics(all_outputs, all_targets)
    # 打印结果
    # print(f'AUC: {auc:.4f}', f'ACC: {acc:.4f}')
    return loss_meter.avg, acc, auc,(acc+auc)/2, writer


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

    # 自动关闭图像
    plt.close()
def adaptive_calculate_metrics(all_outputs, all_targets_onehot,threshold=None, valdata='shanxi_test',num_classes=2, plot_roc=True, save_fpr_tpr=True):
    # 将输出通过 Sigmoid 激活函数转换为概率
    # # probabilities = torch.sigmoid(all_outputs).cpu().numpy()
    # probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    # # probabilities = all_outputs.cpu().numpy()
    # all_targets = all_targets_onehot.cpu().numpy()
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
            # 自动关闭图像
            plt.close()

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
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images,masks,targets,_) in enumerate(data_loader):
        images = images.to(device)
        labels = targets.to(device)
        masks=masks.to(device)
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred,_ = model(images*masks)
        sick_outputs = pred[:, :3].sum(dim=-1)
        health_outputs = pred[:, 3:].sum(dim=-1)
        pred = torch.stack([sick_outputs, health_outputs], dim=1)
        if args.model_name == "dinov2":
            pred = pred[0]
        # 收集输出和目标
        # all_outputs.append(pred)
        # all_targets.append(labels)
        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        loss = nn.CrossEntropyLoss()(pred, labels)

        loss_meter.update(loss.item(), labels.size(0))
        # acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    # all_outputs = torch.cat(all_outputs)
    # all_targets = torch.cat(all_targets)
    # 计算整体指标
    # 计算整体指标
    accuracy, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(all_outputs,
                                                                                                all_targets,
                                                                                                threshold=threshold,
                                                                                                valdata=valdata,
                                                                                                plot_roc=True,
                                                                                                save_fpr_tpr=True)


    # 打印结果
    print(f'Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
          f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, '
          f'F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return accuracy, sensitivity, specificity, auc, f1, threshold_mcc

if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    lrs = [5e-5]
    batch_sizes = [16]
    model_name = ['convnext']
    for kk in range(len(lrs)):
        for jj in range(len(model_name)):
            # shanxi_test_avgs=[]
            # shanxi_test_accs = []
            # shanxi_test_aucs=[]
            # shanxi_test_sens = []
            # shanxi_test_spec = []
            #
            # guizhou_test_avgs=[]
            # guizhou_test_accs = []
            # guizhou_test_aucs = []
            # guizhou_test_sens = []
            # guizhou_test_spec = []
            #
            # val_aucs=[]
            #
            # # for ii in range(len(seeds)):
            # for ii in range(1):
            #     ii+=1
            #     avgs = []
            #     lr = lrs[kk]
            #     args = parse_option()
            #     args.batch_size = batch_sizes[jj]
            #     args.base_lr = lr
            #     args.experi = str(ii)
            #     args.model_name = model_name[jj]
            #     args.sub_weight=1
            #     args.cam_weight = 0
            #     config = get_config(args)
            #     config.OUTPUT = os.path.join(args.output, args.model_name+'_new_wmask_wsubcls',
            #                                  str(args.experi) + 'experi_' + args.model_name + '_lr' + str(args.base_lr))
            #     config.TRAIN.EPOCHS = 60
            #     config.freeze()
            #
            #     os.makedirs(config.OUTPUT, exist_ok=True)
            #     logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
            #
            #     path = os.path.join(config.OUTPUT, "config.json")
            #     with open(path, "w") as f:
            #         f.write(config.dump())
            #     logger.info(f"Full config saved to {path}")
            #
            #     # print config
            #     logger.info(config.dump())
            #     logger.info(json.dumps(vars(args)))
            #
            #     mcc_threshold, guizhou_threshold, shanxi_val_auc, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity = main(config)
            #     shanxi_avgs=(shanxi_test_auc+shanxi_test_accuracy+shanxi_test_sensitivity+shanxi_test_specificity)/4
            #     shanxi_test_avgs.append(shanxi_avgs)
            #     shanxi_test_accs.append(shanxi_test_accuracy)
            #     shanxi_test_aucs.append(shanxi_test_auc)
            #     shanxi_test_sens.append(shanxi_test_sensitivity)
            #     shanxi_test_spec.append(shanxi_test_specificity)
            #
            #     guizhou_avgs = (guizhou_test_auc + guizhou_test_accuracy + guizhou_test_sensitivity + guizhou_test_specificity) / 4
            #     guizhou_test_avgs.append(guizhou_avgs)
            #     guizhou_test_accs.append(guizhou_test_accuracy)
            #     guizhou_test_aucs.append(guizhou_test_auc)
            #     guizhou_test_sens.append(guizhou_test_sensitivity)
            #     guizhou_test_spec.append(guizhou_test_specificity)
            #
            #     val_aucs.append(shanxi_val_auc)
            #     # 移除之前添加的处理器
            #     for handler in logger.handlers[:]:  # 循环遍历处理器的副本
            #         logger.removeHandler(handler)  # 移除处理器
            #     gc.collect()
            #     folder_path = config.OUTPUT.split(config.OUTPUT.split('/')[-1])[0]
            #     if not os.path.exists(folder_path):
            #         os.makedirs(folder_path)
            #     file_name = 'result.txt'
            #     file_path = os.path.join(folder_path, file_name)
            #     with open(file_path, 'a', encoding='utf-8') as file:
            #         re = f'model_ema_best_acc_shanxi_val {args.experi}experi_best_result: threshold: val {mcc_threshold}, best {guizhou_threshold}, lr: {args.base_lr}, val_auc: {shanxi_val_auc}, ' \
            #              f'shanxi_test_avg: {shanxi_avgs}, guizhou_test_avg: {guizhou_avgs}, shanxi_test_auc: {shanxi_test_auc}, shanxi_test_acc: {shanxi_test_accuracy}, shanxi_test_sensitivity: {shanxi_test_sensitivity}, shanxi_test_specificity: {shanxi_test_specificity}' \
            #              f', guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_sensitivity: {guizhou_test_sensitivity}, guizhou_test_specificity: {guizhou_test_specificity}\n'
            #         file.write(re)
            #
            # # 将列表转换为 NumPy 数组
            # shanxi_test_avgs_array = np.array(shanxi_test_avgs)
            # # 计算均值
            # mean_shanxi_test_avgs_value = np.mean(shanxi_test_avgs_array)
            # # 计算标准差
            # std_shanxi_test_avgs_value = np.std(shanxi_test_avgs_array)
            # # 将列表转换为 NumPy 数组
            # shanxi_test_aucs_array = np.array(shanxi_test_aucs)
            # # 计算均值
            # mean_shanxi_test_aucs_value = np.mean(shanxi_test_aucs_array)
            # # 计算标准差
            # std_shanxi_test_aucs_value = np.std(shanxi_test_aucs_array)
            # # 将列表转换为 NumPy 数组
            # shanxi_test_accs_array = np.array(shanxi_test_accs)
            # # 计算均值
            # mean_shanxi_test_accs_value = np.mean(shanxi_test_accs_array)
            # # 计算标准差
            # std_shanxi_test_accs_value = np.std(shanxi_test_accs_array)
            # # 将列表转换为 NumPy 数组
            # shanxi_test_sens_array = np.array(shanxi_test_sens)
            # # 计算均值
            # mean_shanxi_test_sens_value = np.mean(shanxi_test_sens_array)
            # # 计算标准差
            # std_shanxi_test_sens_value = np.std(shanxi_test_sens_array)
            # # 将列表转换为 NumPy 数组
            # shanxi_test_spec_array = np.array(shanxi_test_spec)
            # # 计算均值
            # mean_shanxi_test_spec_value = np.mean(shanxi_test_spec_array)
            # # 计算标准差
            # std_shanxi_test_spec_value = np.std(shanxi_test_spec_array)
            #
            # # 将列表转换为 NumPy 数组
            # guizhou_test_avgs_array = np.array(guizhou_test_avgs)
            # # 计算均值
            # mean_guizhou_test_avgs_value = np.mean(guizhou_test_avgs_array)
            # # 计算标准差
            # std_guizhou_test_avgs_value = np.std(guizhou_test_avgs_array)
            # # 将列表转换为 NumPy 数组
            # guizhou_test_aucs_array = np.array(guizhou_test_aucs)
            # # 计算均值
            # mean_guizhou_test_aucs_value = np.mean(guizhou_test_aucs_array)
            # # 计算标准差
            # std_guizhou_test_aucs_value = np.std(guizhou_test_aucs_array)
            # # 将列表转换为 NumPy 数组
            # guizhou_test_accs_array = np.array(guizhou_test_accs)
            # # 计算均值
            # mean_guizhou_test_accs_value = np.mean(guizhou_test_accs_array)
            # # 计算标准差
            # std_guizhou_test_accs_value = np.std(guizhou_test_accs_array)
            # # 将列表转换为 NumPy 数组
            # guizhou_test_sens_array = np.array(guizhou_test_sens)
            # # 计算均值
            # mean_guizhou_test_sens_value = np.mean(guizhou_test_sens_array)
            # # 计算标准差
            # std_guizhou_test_sens_value = np.std(guizhou_test_sens_array)
            # # 将列表转换为 NumPy 数组
            # guizhou_test_spec_array = np.array(guizhou_test_spec)
            # # 计算均值
            # mean_guizhou_test_spec_value = np.mean(guizhou_test_spec_array)
            # # 计算标准差
            # std_guizhou_test_spec_value = np.std(guizhou_test_spec_array)
            #
            #
            # # 将列表转换为 NumPy 数组
            # val_aucs_array = np.array(val_aucs)
            # # 计算均值
            # mean_val_aucs_value = np.mean(val_aucs_array)
            # # 计算标准差
            # std_val_aucs_value = np.std(val_aucs_array)
            #
            #
            # with open(file_path, 'a', encoding='utf-8') as file:
            #     file.write(f"Shanxi Val AUC: {mean_val_aucs_value:.2f}±{std_val_aucs_value:.2f}\n")
            #     file.write(f"Shanxi Test Avg: {mean_shanxi_test_avgs_value:.2f}±{std_shanxi_test_avgs_value:.2f}\n")
            #     file.write(f"Guizhou Test Avg: {mean_guizhou_test_avgs_value:.2f}±{std_guizhou_test_avgs_value:.2f}\n")
            #     file.write(f"Shanxi Test AUC: {mean_shanxi_test_aucs_value:.2f}±{std_shanxi_test_aucs_value:.2f}\n")
            #     file.write(f"Shanxi Test ACC: {mean_shanxi_test_accs_value:.2f}±{std_shanxi_test_accs_value:.2f}\n")
            #     file.write(f"Shanxi Test Sensitivity: {mean_shanxi_test_sens_value:.2f}±{std_shanxi_test_sens_value:.2f}\n")
            #     file.write(f"Shanxi Test Specificity: {mean_shanxi_test_spec_value:.2f}±{std_shanxi_test_spec_value:.2f}\n")
            #     file.write(f"Guizhou Test AUC: {mean_guizhou_test_aucs_value:.2f}±{std_guizhou_test_aucs_value:.2f}\n")
            #     file.write(f"Guizhou Test ACC: {mean_guizhou_test_accs_value:.2f}±{std_guizhou_test_accs_value:.2f}\n")
            #     file.write(f"Guizhou Test Sensitivity: {mean_guizhou_test_sens_value:.2f}±{std_guizhou_test_sens_value:.2f}\n")
            #     file.write(f"Guizhou Test Specificity: {mean_guizhou_test_spec_value:.2f}±{std_guizhou_test_spec_value:.2f}\n")



            shanxi_test_avgs = []
            shanxi_test_accs = []
            shanxi_test_aucs = []
            shanxi_test_sens = []
            shanxi_test_spec = []

            guizhou_test_avgs = []
            guizhou_test_accs = []
            guizhou_test_aucs = []
            guizhou_test_sens = []
            guizhou_test_spec = []

            val_aucs = []

            # for ii in range(len(seeds)):
            for ii in range(1):
                ii += 1
                avgs = []
                lr = lrs[kk]
                args = parse_option()
                args.batch_size = batch_sizes[jj]
                args.base_lr = lr
                args.experi = str(ii)
                args.model_name = model_name[jj]
                args.sub_weight = 1
                args.cam_weight=1
                config = get_config(args)
                config.OUTPUT = os.path.join(args.output, args.model_name + '_random_new_wmask_wsubcls_wcamloss',
                                             str(args.experi) + 'experi_' + args.model_name + '_lr' + str(args.base_lr))
                config.TRAIN.EPOCHS = 50
                config.TRAIN.WARMUP_EPOCHS = 0
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

                mcc_threshold, guizhou_threshold, shanxi_val_auc, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity = main(
                    config)
                shanxi_avgs = (
                                          shanxi_test_auc + shanxi_test_accuracy + shanxi_test_sensitivity + shanxi_test_specificity) / 4
                shanxi_test_avgs.append(shanxi_avgs)
                shanxi_test_accs.append(shanxi_test_accuracy)
                shanxi_test_aucs.append(shanxi_test_auc)
                shanxi_test_sens.append(shanxi_test_sensitivity)
                shanxi_test_spec.append(shanxi_test_specificity)

                guizhou_avgs = (
                                           guizhou_test_auc + guizhou_test_accuracy + guizhou_test_sensitivity + guizhou_test_specificity) / 4
                guizhou_test_avgs.append(guizhou_avgs)
                guizhou_test_accs.append(guizhou_test_accuracy)
                guizhou_test_aucs.append(guizhou_test_auc)
                guizhou_test_sens.append(guizhou_test_sensitivity)
                guizhou_test_spec.append(guizhou_test_specificity)

                val_aucs.append(shanxi_val_auc)
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
                    re = f'model_ema_best_acc_shanxi_val {args.experi}experi_best_result: threshold: val {mcc_threshold}, best {guizhou_threshold}, lr: {args.base_lr}, val_auc: {shanxi_val_auc}, ' \
                         f'shanxi_test_avg: {shanxi_avgs}, guizhou_test_avg: {guizhou_avgs}, shanxi_test_auc: {shanxi_test_auc}, shanxi_test_acc: {shanxi_test_accuracy}, shanxi_test_sensitivity: {shanxi_test_sensitivity}, shanxi_test_specificity: {shanxi_test_specificity}' \
                         f', guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_sensitivity: {guizhou_test_sensitivity}, guizhou_test_specificity: {guizhou_test_specificity}\n'
                    file.write(re)

            # 将列表转换为 NumPy 数组
            shanxi_test_avgs_array = np.array(shanxi_test_avgs)
            # 计算均值
            mean_shanxi_test_avgs_value = np.mean(shanxi_test_avgs_array)
            # 计算标准差
            std_shanxi_test_avgs_value = np.std(shanxi_test_avgs_array)
            # 将列表转换为 NumPy 数组
            shanxi_test_aucs_array = np.array(shanxi_test_aucs)
            # 计算均值
            mean_shanxi_test_aucs_value = np.mean(shanxi_test_aucs_array)
            # 计算标准差
            std_shanxi_test_aucs_value = np.std(shanxi_test_aucs_array)
            # 将列表转换为 NumPy 数组
            shanxi_test_accs_array = np.array(shanxi_test_accs)
            # 计算均值
            mean_shanxi_test_accs_value = np.mean(shanxi_test_accs_array)
            # 计算标准差
            std_shanxi_test_accs_value = np.std(shanxi_test_accs_array)
            # 将列表转换为 NumPy 数组
            shanxi_test_sens_array = np.array(shanxi_test_sens)
            # 计算均值
            mean_shanxi_test_sens_value = np.mean(shanxi_test_sens_array)
            # 计算标准差
            std_shanxi_test_sens_value = np.std(shanxi_test_sens_array)
            # 将列表转换为 NumPy 数组
            shanxi_test_spec_array = np.array(shanxi_test_spec)
            # 计算均值
            mean_shanxi_test_spec_value = np.mean(shanxi_test_spec_array)
            # 计算标准差
            std_shanxi_test_spec_value = np.std(shanxi_test_spec_array)

            # 将列表转换为 NumPy 数组
            guizhou_test_avgs_array = np.array(guizhou_test_avgs)
            # 计算均值
            mean_guizhou_test_avgs_value = np.mean(guizhou_test_avgs_array)
            # 计算标准差
            std_guizhou_test_avgs_value = np.std(guizhou_test_avgs_array)
            # 将列表转换为 NumPy 数组
            guizhou_test_aucs_array = np.array(guizhou_test_aucs)
            # 计算均值
            mean_guizhou_test_aucs_value = np.mean(guizhou_test_aucs_array)
            # 计算标准差
            std_guizhou_test_aucs_value = np.std(guizhou_test_aucs_array)
            # 将列表转换为 NumPy 数组
            guizhou_test_accs_array = np.array(guizhou_test_accs)
            # 计算均值
            mean_guizhou_test_accs_value = np.mean(guizhou_test_accs_array)
            # 计算标准差
            std_guizhou_test_accs_value = np.std(guizhou_test_accs_array)
            # 将列表转换为 NumPy 数组
            guizhou_test_sens_array = np.array(guizhou_test_sens)
            # 计算均值
            mean_guizhou_test_sens_value = np.mean(guizhou_test_sens_array)
            # 计算标准差
            std_guizhou_test_sens_value = np.std(guizhou_test_sens_array)
            # 将列表转换为 NumPy 数组
            guizhou_test_spec_array = np.array(guizhou_test_spec)
            # 计算均值
            mean_guizhou_test_spec_value = np.mean(guizhou_test_spec_array)
            # 计算标准差
            std_guizhou_test_spec_value = np.std(guizhou_test_spec_array)

            # 将列表转换为 NumPy 数组
            val_aucs_array = np.array(val_aucs)
            # 计算均值
            mean_val_aucs_value = np.mean(val_aucs_array)
            # 计算标准差
            std_val_aucs_value = np.std(val_aucs_array)

            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(f"Shanxi Val AUC: {mean_val_aucs_value:.2f}±{std_val_aucs_value:.2f}\n")
                file.write(f"Shanxi Test Avg: {mean_shanxi_test_avgs_value:.2f}±{std_shanxi_test_avgs_value:.2f}\n")
                file.write(f"Guizhou Test Avg: {mean_guizhou_test_avgs_value:.2f}±{std_guizhou_test_avgs_value:.2f}\n")
                file.write(f"Shanxi Test AUC: {mean_shanxi_test_aucs_value:.2f}±{std_shanxi_test_aucs_value:.2f}\n")
                file.write(f"Shanxi Test ACC: {mean_shanxi_test_accs_value:.2f}±{std_shanxi_test_accs_value:.2f}\n")
                file.write(
                    f"Shanxi Test Sensitivity: {mean_shanxi_test_sens_value:.2f}±{std_shanxi_test_sens_value:.2f}\n")
                file.write(
                    f"Shanxi Test Specificity: {mean_shanxi_test_spec_value:.2f}±{std_shanxi_test_spec_value:.2f}\n")
                file.write(f"Guizhou Test AUC: {mean_guizhou_test_aucs_value:.2f}±{std_guizhou_test_aucs_value:.2f}\n")
                file.write(f"Guizhou Test ACC: {mean_guizhou_test_accs_value:.2f}±{std_guizhou_test_accs_value:.2f}\n")
                file.write(
                    f"Guizhou Test Sensitivity: {mean_guizhou_test_sens_value:.2f}±{std_guizhou_test_sens_value:.2f}\n")
                file.write(
                    f"Guizhou Test Specificity: {mean_guizhou_test_spec_value:.2f}±{std_guizhou_test_spec_value:.2f}\n")
