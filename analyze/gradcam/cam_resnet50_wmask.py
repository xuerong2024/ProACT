import os
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
from pytorch_grad_cam import GradCAM, \
        ScoreCAM, \
        GradCAMPlusPlus, \
        AblationCAM, \
        XGradCAM, \
        EigenCAM, \
        EigenGradCAM, \
        LayerCAM, \
        FullGrad
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')
# 添加自定义路径到全局模块搜索路径中
sys.path.append('/disk3/wjr/workspace/sec_nejm/nejm_baseline_wjr/')
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')


torch.set_num_threads(1)


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return
def reshape_transform_swin(tensor, height=14, width=14):
    # 去掉cls token
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    b,hw,c=tensor.shape
    h=int(np.sqrt(hw))

    result = tensor.reshape(b,
    h, h, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.permute(0,3,1,2)
    return result
def reshape_transform_dino(tensor, height=14, width=14):
    # 去掉cls token
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    # tensor=torch.cat((tensor[:, :98, :],tensor[:, 99:, :]),dim=1)
    tensor = tensor[:, 1:, :]
    b,hw,c=tensor.shape
    h=int(np.sqrt(hw))

    result = tensor.reshape(b,
    h, h, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.permute(0,3,1,2)
    return result
def reshape_transform(tensor):
    # 去掉cls token
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    # print(tensor.shape)
    return tensor
def reshape_transform_mambaout(tensor):
    # 去掉cls token
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    # print(tensor.shape)
    return tensor.permute(0,3,1,2)
if __name__ == '__main__':
    import shutil
    from mmpretrain import get_model
    from model_cls.convnext import *
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pathlib import Path
    from data import transforms as d_transform
    from utils.cloc import *
    from utils.datasets_pneum import *
    from model_cls.pka2net import *
    from model_cls.pcam_densenet import *
    from model_cls.dmalnet.fusionnet import fusion_net
    from model_cls.mambaout import MambaOut

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
    parser.add_argument('--output',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/resnet50/wmask_nosubcls_nodropout2/resnet50_lr8e-05/',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_nosubcls_nodropout2/convnext_lr5e-05/',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/swintiny/wmask_nosubcls_nodropout2/swintiny_lr0.0001/',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/dinov2/wmask_nosubcls_nodropout2/dinov2_lr3e-05_testall/',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/mambaout/wmask_nosubcls_nodropout2/mambaout_lr3e-05_testall/',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool',
                        # default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/pcam/wmask_nosubcls_nodropout2/pcam_lr3e-05',
                        default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/',
                        type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    # parser.add_argument('--output',
    #                     default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250610/convnext_new_wmask_nosubcls/',
    #                     type=str, metavar='PATH',
    #                     help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
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

    parser.add_argument('--data_root', type=str, default='/disk3/wjr/dataset/nejm/shanxidataset/',
                        help='The path of dataset')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument("--threshold", type=float, default=0.5, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()
    args.experi='1'
    args.base_lr=8e-5
    args.patch_size=16
    # args.model_name = "resnet50"
    # args.model_name = "convnext"
    # args.model_name = "swintiny"
    # args.model_name = "dinov2"
    # args.model_name = "mambaout"
    # args.model_name = "DMALNET"
    # args.model_name = "pcam"
    # args.model_name = "our"
    # args.model_name = "convnext_local"
    # args.model_name = "convnext_local_tree"
    # args.model_name = "convnext_local_tree_cam"
    # args.model_name = "convnext_local_tree_cloc"
    args.model_name = "convnext_local_tree_cam_align"
    outputs= {"resnet50": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/resnet50/wmask_nosubcls_nodropout2/resnet50_lr8e-05/',
              "convnext": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_nosubcls_nodropout2/convnext_lr5e-05/',
              "swintiny": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/swintiny/wmask_nosubcls_nodropout2/swintiny_lr0.0001/',
              "dinov2": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/dinov2/wmask_nosubcls_nodropout2/dinov2_lr3e-05_testall/',
              "mambaout": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/mambaout/wmask_nosubcls_nodropout2/mambaout_lr3e-05_testall/',
              "DMALNET": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool',
              "pcam": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/pcam/wmask_nosubcls_nodropout2/pcam_lr3e-05',
              "our": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/',
              "convnext_local":'/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_w5&2subcls_nodropout/ce_convnext_lr5e-05',
              "convnext_local_tree": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnextglobal512_local256_treemodel_5cls_2cls_2/1experi_convnext_lr0.0001',
              "convnext_local_tree_cam": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/ablation/cam_cloc_align_weight/0.5_cam_0_camalign_0_cloc_lr8e-05',
              "convnext_local_tree_cloc": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/ablation/cam_cloc_align_weight/0_cam_0_camalign_0.3_cloc_lr8e-05',
              "convnext_local_tree_cam_align": '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/ablation/cam_cloc_align_weight/0_cam_0.5_camalign_0_cloc_lr8e-05',
              }

    # 获取resnet50对应的权重路径
    if args.model_name in outputs:
        args.output = outputs[args.model_name]
        print(f"找到 {args.model_name} 的权重路径: {args.output}")
    if args.model_name == "convnext":
        model = convnexttiny_org(pretrained=True, drop_path_rate=0)
        model.head.fc = nn.Linear(768, 2)
    if args.model_name == "convnext_local":
        model = convnexttiny_2_5cls(pretrained=True, drop_path_rate=0)
        model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "our" or "convnext_local_tree" in args.model_name:
        model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
        model.head.fc = nn.Linear(768, 2)
    elif args.model_name=='resnet50':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        model = get_model('resnet50_8xb32_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(2048, 2)
    elif args.model_name == "swintiny":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        model.head.fc = nn.Linear(768, 2, bias=True)

    elif args.model_name == "dinov2":
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'
        model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=pretrained_cfg)
        model.backbone.ln1 = nn.Linear(384, 2, bias=True)
    elif args.model_name == 'mambaout':
        model = MambaOut(num_classes=2,
                         depths=[3, 3, 9, 3],
                         dims=[96, 192, 384, 576])
    elif args.model_name == 'PKA2_Net':
        model = PKA2_Net(n_class=2)
    elif args.model_name == 'pcam':
        model = pcam_dense121(num_classes=[1, 1])
    elif args.model_name == 'medmamba':
        from model_cls.medmamba import VSSM as medmamba
        model = medmamba(num_classes=2)
    elif args.model_name == 'DMALNET':
        model = fusion_net(pretrained_pth='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool')

    model.to(device)
    for name, param in model.named_parameters():
        print(name)
    org_output_dir = args.output
    if args.model_name != 'DMALNET':
        checkpoint = torch.load(os.path.join(org_output_dir, "model_ema_best_auc_shanxi_val.pth"), map_location='cpu')
        if args.model_name == "dinov2" or args.model_name == "our" or args.model_name == "convnext_local_tree_cam_align":
            checkpoint = torch.load(os.path.join(org_output_dir, "model_ema_best_acc_shanxi_val.pth"),
                                    map_location='cpu')
        if args.model_name == "convnext_local_tree_cam" or args.model_name == "convnext_local_tree_cloc":
            checkpoint = torch.load(os.path.join(org_output_dir, "model_ema_best_auc_guizhou_test.pth"),
                                    map_location='cpu')
            # checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint_best.pth.tar"), map_location='cpu')
        state_dict = checkpoint['model']
        # if args.model_name == "convnext":
        #     state_dict = {"backbone."+k: v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)


    # print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))


    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    pil2tensor_transfo = d_transform.Compose([d_transform.Resize((512, 512)), d_transform.ToTensor(),
                                              d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
    if args.model_name == 'DMALNET':
        dataset_train = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_mask_img_1024',
                                           args.data_root + 'seg_rec_mask_1024',
                                           txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick.txt',
                                           data_transform=pil2tensor_transfo)
        dataset_val = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_mask_img_1024', args.data_root + 'seg_rec_mask_1024',
                                           txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
                                           data_transform=pil2tensor_transfo)
        dataset_test = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_img_1024',
                                           '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                           txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_sick_one.txt',
                                           data_transform=pil2tensor_transfo)
    else:
        dataset_train = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + 'seg_rec_mask_1024',
                                           txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick.txt',
                                           data_transform=pil2tensor_transfo)
        dataset_val = Shanxi_wmask_Dataset(args.data_root + 'seg_rec_img_1024', args.data_root + 'seg_rec_mask_1024',
                                       txtpath=args.data_root + 'stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val.txt',
                                       data_transform=pil2tensor_transfo)
        dataset_test = Shanxi_wmask_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                            '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                            txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_sick_one.txt',
                                            data_transform=pil2tensor_transfo)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    output_dir = org_output_dir + '/analyze/shanxivalGradCAM'+ '/layers_1/'
    print(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.model_name == 'resnet50':
        cam = GradCAM(model=model,
                      target_layers=[model.backbone.layer4[-1]],
                      reshape_transform=reshape_transform)
    elif args.model_name == 'convnext':
        cam = GradCAM(model=model,
                      target_layers=[model.stages[-1]],
                      reshape_transform=reshape_transform)
    elif args.model_name == 'our' or "convnext_local" in args.model_name:
        cam = GradCAM(model=model,
                      target_layers=[model.stages[-1]],
                      reshape_transform=reshape_transform)
    elif args.model_name == 'mambaout':
        cam = GradCAM(model=model,
                      target_layers=[model.stages[-1]],
                      reshape_transform=reshape_transform_mambaout)
    elif args.model_name == "dinov2":
        cam = GradCAM(model=model,
                      target_layers=[model.backbone.layers[-1].ln2],
                      # target_layers=[model.blocks[-8].norm1],
                      # use_cuda=True,
                      reshape_transform=reshape_transform_dino)
    elif args.model_name == 'swintiny':
        cam = GradCAM(model=model,
                      target_layers=[model.backbone.stages[-1].blocks[-1].norm1],
                      reshape_transform=reshape_transform_swin)
    elif args.model_name == 'DMALNET':
        cam = GradCAM(model=model,
                      target_layers=[model.second_model.layer4[-1]],
                      reshape_transform=reshape_transform)
    elif args.model_name == 'pcam':
        cam = GradCAM(model=model,
                      target_layers=[model.backbone.features.denseblock4[-1]],
                      reshape_transform=reshape_transform)
    for idx, (images, masks, targets, img_names) in enumerate(data_loader_val):
        # rand_module.randomize()
        if args.model_name == 'DMALNET':
            img = images
        else:
            img = (images * masks)
        # images = images.to(device)
        labels = targets.to(device)
        batch_tensor = img[0, ...].permute(1, 2, 0)
        # 反归一化：(x * std) + mean
        denorm_batch = batch_tensor * std + mean
        denorm_batch =denorm_batch*masks[0, ...].permute(1, 2, 0)
        denorm_batch = torch.clamp(denorm_batch, 0, 1)
        rgb_img = np.array(denorm_batch)
        imgname = img_names[0]
        # print('processing ', imgname)
        # output = model(img.to(device))
        output = model(img.to(device))
        if args.model_name == "dinov2":
            output = output[1]


        aa = output.squeeze(0).argmax().item()
        if aa == 1:
            pred_name = 'Health'
            # grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
            #     0)])
            # grayscale_cam1 = cam(input_tensor=img.to(device))
            grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                1)])
            grayscale_cam1 = grayscale_cam1[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
        else:
            pred_name = 'Sick'
            grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                0)])
            # grayscale_cam0 = cam(input_tensor=img.to(device))
            grayscale_cam0 = grayscale_cam0[0, :]
            # 将 grad-cam 的输出叠加到原始图像上
            visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)

    output_dir = org_output_dir + '/analyze/guizhoutestGradCAM' + '/layers_1/'
    print(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, (images, masks, targets, img_names) in enumerate(data_loader_test):
        # rand_module.randomize()
        if args.model_name == 'DMALNET':
            img = images
        else:
            img = (images * masks)
        # images = images.to(device)
        labels = targets.to(device)
        batch_tensor = img[0, ...].permute(1, 2, 0)
        # 反归一化：(x * std) + mean
        denorm_batch = batch_tensor * std + mean
        denorm_batch =denorm_batch*masks[0, ...].permute(1, 2, 0)
        denorm_batch = torch.clamp(denorm_batch, 0, 1)
        rgb_img = np.array(denorm_batch)
        imgname = img_names[0]
        # print('processing ', imgname)
        # output = model(img.to(device))
        output = model(img.to(device))
        if args.model_name == "dinov2":
            output = output[1]


        aa = output.squeeze(0).argmax().item()
        if aa == 1:
            pred_name = 'Health'
            # grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
            #     0)])
            # grayscale_cam1 = cam(input_tensor=img.to(device))
            grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                1)])
            grayscale_cam1 = grayscale_cam1[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
        else:
            pred_name = 'Sick'
            grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                0)])
            # grayscale_cam0 = cam(input_tensor=img.to(device))
            grayscale_cam0 = grayscale_cam0[0, :]
            # 将 grad-cam 的输出叠加到原始图像上
            visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)

    output_dir = org_output_dir + '/analyze/shanxitrainGradCAM' + '/layers_1/'
    print(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, (images, masks, targets, img_names) in enumerate(data_loader_train):
        # rand_module.randomize()
        if args.model_name == 'DMALNET':
            img = images
        else:
            img = (images * masks)
        # images = images.to(device)
        labels = targets.to(device)
        batch_tensor = img[0, ...].permute(1, 2, 0)
        # 反归一化：(x * std) + mean
        denorm_batch = batch_tensor * std + mean
        denorm_batch = denorm_batch * masks[0, ...].permute(1, 2, 0)
        denorm_batch = torch.clamp(denorm_batch, 0, 1)
        rgb_img = np.array(denorm_batch)
        imgname = img_names[0]
        # print('processing ', imgname)
        # output = model(img.to(device))
        output = model(img.to(device))
        if args.model_name == "dinov2":
            output = output[1]

        aa = output.squeeze(0).argmax().item()
        if aa == 1:
            pred_name = 'Health'
            # grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
            #     0)])
            # grayscale_cam1 = cam(input_tensor=img.to(device))
            grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                1)])
            grayscale_cam1 = grayscale_cam1[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
        else:
            pred_name = 'Sick'
            grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                0)])
            # grayscale_cam0 = cam(input_tensor=img.to(device))
            grayscale_cam0 = grayscale_cam0[0, :]
            # 将 grad-cam 的输出叠加到原始图像上
            visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
            # 保存可视化结果
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            cv2.imwrite(os.path.join(output_dir, str(idx) + imgname.split('.png')[0].split('_')[
                0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)

    print('saving cam images in ', output_dir)