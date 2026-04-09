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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

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

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    # if isinstance(tensor, tuple):
    #     tensor=tensor[0]
    # tensor=torch.cat((tensor[:, :98, :],tensor[:, 99:, :]),dim=1)
    # # tensor = tensor[:, 1:, :]
    # b,hw,c=tensor.shape
    # h=int(np.sqrt(hw))
    #
    # result = tensor.reshape(b,
    # h, h, tensor.size(2))

    # 将通道维度放到第一个位置
    # result = result.permute(0,3,1,2)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


if __name__ == '__main__':
    import shutil
    from mmpretrain import get_model
    from model_cls.convnext import *
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pathlib import Path
    from model_cls.vmambav2.vmamba import *
    from data import transforms
    from utils.cloc import *
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
                        default='/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/gradcam/calgm/org/',
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
    args.model_name = "slgms"
    # args.txtpath = '/disk3/wjr/dataset/nejm/shanxidataset/sick_health_5fold_txt/fold1_train_sick_subregion.txt'

    # args.txtpath = '/disk3/wjr/dataset/nejm/shanxidataset/sick_health_5fold_txt/fold1_test.txt'
    # args.txtpath = '/disk3/wjr/dataset/nejm/cam_shanxi_test1.txt'
    args.txtpath = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/shanxi.txt'
    # args.txtpath = '/disk3/wjr/dataset/nejm/cam_guihzou_one.txt'
    # args.data_root = '/disk3/wjr/dataset/nejm/guizhoudataset/'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
    # model = convnexttiny_hiera_tree_pixelshuffleattn(pretrained=True)
    # model.head.fc = nn.Linear(768, 2)
    # learnable_map = [
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    # ]
    # model.margin_criterion = OrdinalContrastiveLoss_mm(
    #     n_classes=5,
    #     device=device,
    #     learnable_map=learnable_map
    # )
    # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
    # model.head.fc = nn.Linear(768, 2)
    # model = convnexttiny_org(pretrained=True)
    # model.head.fc = nn.Linear(768, 2)
    # model = convnexttiny(pretrained=True)
    # model.head.fc = nn.Linear(768, 2)
    if args.model_name == "convnext":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        # model = get_model('convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k', pretrained=pretrained_cfg)
        # model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_org(pretrained=True, drop_path_rate=0.)
        # model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_pcam_hiera_tree(pretrained=True, drop_path_rate=0)
        model = convnexttiny_hiera_tree(pretrained=True, drop_path_rate=0)
        model.head.fc = nn.Linear(768, 2)
        # model = convnexttiny_2_5cls(pretrained=True, drop_path_rate=0)
        # model.head.fc = nn.Linear(768, 2)
    elif args.model_name == "slgms":
        model=DINO_VSSM()
    elif args.model_name == "convnext_fpn_x16":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        model = convnexttiny_fpn_x16_fine_global(pretrained=True)
    elif args.model_name == "convnext_fpn_x8":
        # pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
        model = convnexttiny_fpn_x8_fine_global(pretrained=True)
    elif args.model_name=='resnet50':
        pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
        model = get_model('resnet50_8xb32_in1k', pretrained=pretrained_cfg, backbone=dict(drop_path_rate=0.))
        # net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.head.fc = nn.Linear(2048, 2)
    # model = convnexttiny_hiera_tree_pixelshuffleattn(pretrained=True)
    # model.head.fc = nn.Linear(768, 2)
    # learnable_map = [
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    #     ['fixed', 0.4],
    # ]
    # model.margin_criterion = OrdinalContrastiveLoss_mm(
    #     n_classes=5,
    #     device=device,
    #     learnable_map=learnable_map
    # )
    model.to(device)
    args.output='/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/gradcam/slgms/org/'
    org_output_dir=args.output
    # org_output_dir=os.path.join(args.output, 'ce_convnext_lr5e-05',)
    # org_output_dir = os.path.join(args.output, '1experi_convnext_lr5e-05', )
    # ckpath='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/model_ema_best_acc_shanxi_val.pth'
    ckpath = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/checkpoint_best.pth.tar'
    state_dict = torch.load(ckpath, map_location='cpu', weights_only=False)['teacher']
    state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

    # ckpath='/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/shanxi_test.pth'
    # state_dict = torch.load(ckpath, map_location='cpu', weights_only=False)['model']
    # msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    # print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    for name, param in model.named_parameters():
        print(name)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # # open image
    if args.data_root is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    else:
        imgs = []
        img_names = []
        rgb_imgs = []
        with open(args.txtpath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            imgs = []
            img_names = []
            rgb_imgs = []
            with open(args.txtpath, 'r', encoding='gbk') as file:
                lines = file.readlines()
            for line in lines:
                # imgname = line.split(':')[0]
                imgname = line.split('\n')[0]
                labelname = imgname.split('_')[0]
                if '.png' in imgname:
                    path = os.path.join(args.data_root + 'seg_rec_mask_img_1024', imgname)
                    maskpath = os.path.join(args.data_root + 'seg_rec_mask_1024', imgname)
                else:
                    path = os.path.join(args.data_root + 'seg_rec_img_1024', imgname + '.png')
                    maskpath = os.path.join(args.data_root + 'seg_rec_mask_1024', imgname + '.png')
                # shutil.copy(path, args.output_dir)
                rgb_img = cv2.imread(path, 1)[:, :, ::-1]
                rgb_img = cv2.resize(rgb_img, (512, 512)) / 255.
                rgb_imgs.append(rgb_img)
                img = Image.open(path)
                img = img.convert('RGB')
                maskimg = Image.open(maskpath)
                img, maskimg = transform(img, maskimg)
                imgs.append(img * maskimg)
                img_names.append(imgname.split('.png')[0])
                del img, maskimg, rgb_img
    # args.output_dir = args.output_dir + '/layer0EigenGradCAM/'

    ffs = [3]
    for ee in range(1):
        for jj in range(len(ffs)):
            ff = ffs[jj]
            if ee == 0:
                args.output_dir = org_output_dir + '/GradCAM_shanxi/layers' + str(ff) + '/'
                print(args.output_dir)
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                cam = GradCAM(model=model,
                              # target_layers=[model.fpn.smooth_convs],
                              # target_layers=[model.globalclassifier[0]],
                              # target_layers=[model.stages[-1]],
                              target_layers=[model.layers[-1].blocks[-2]],
                              # target_layers=[model.backbone.stages[-1]],
                              # target_layers=[model.blocks[-8].norm1],
                              # use_cuda=True,
                              reshape_transform=reshape_transform)
            for ii in range(len(imgs)):
                img = imgs[ii]
                rgb_img = rgb_imgs[ii]
                imgname = img_names[ii]
                print(ii, ': ',img_names[ii])
                # make the image divisible by the patch size
                w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
                img = img[:, :w, :h].unsqueeze(0)

                w_featmap = img.shape[-2] // args.patch_size
                h_featmap = img.shape[-1] // args.patch_size
                # output = model(img.to(device))
                output = model(img.to(device))
                pred_name = 'Sick'
                grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                    0)])
                # grayscale_cam0 = cam(input_tensor=img.to(device))
                grayscale_cam0 = grayscale_cam0[0, :]
                # 将 grad-cam 的输出叠加到原始图像上
                visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
                # 保存可视化结果
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                    0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
                # aa = output.squeeze(0).argmax().item()
                # if aa == 1:
                #     pred_name = 'Health'
                #     # grayscale_cam1 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                #     #     1)])
                #     grayscale_cam1 = cam(input_tensor=img.to(device))
                #     grayscale_cam1 = grayscale_cam1[0, :]
                #     visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
                #     # 保存可视化结果
                #     cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                #     cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                #         0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
                # else:
                #     pred_name = 'Sick'
                #     # grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
                #     #     0)])
                #     grayscale_cam0 = cam(input_tensor=img.to(device))
                #     grayscale_cam0 = grayscale_cam0[0, :]
                #     # 将 grad-cam 的输出叠加到原始图像上
                #     visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
                #     # 保存可视化结果
                #     cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                #     cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                #         0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
                ii += 1




    # args.output = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/gradcam/slgms/guizhou_medtta/'
    # org_output_dir = args.output
    # # org_output_dir=os.path.join(args.output, 'ce_convnext_lr5e-05',)
    # # org_output_dir = os.path.join(args.output, '1experi_convnext_lr5e-05', )
    # # ckpath = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/checkpoint_best.pth.tar'
    # # state_dict = torch.load(ckpath, map_location='cpu', weights_only=False)['teacher']
    # # state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
    # # msg = model.load_state_dict(state_dict, strict=False)
    # # print(msg)
    #
    # ckpath='/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/guizhou_test.pth'
    # state_dict = torch.load(ckpath, map_location='cpu', weights_only=False)['model']
    # msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    # args.txtpath = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/guizhou.txt'
    # # open image
    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    # if args.data_root is None:
    #     # user has not specified any image - we use our own image
    #     print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
    #     print("Since no image path have been provided, we take the first image in our paper.")
    #     response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
    #     img = Image.open(BytesIO(response.content))
    #     img = img.convert('RGB')
    # else:
    #     imgs = []
    #     img_names = []
    #     rgb_imgs = []
    #     with open(args.txtpath, 'r', encoding='gbk') as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             imgname = line.split('\n')[0]
    #             labelname = imgname.split('_')[0]
    #             if '.png' in imgname:
    #                 path = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_img_1024', imgname)
    #                 path1 = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024', imgname)
    #                 maskpath = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024', imgname)
    #             else:
    #                 path = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_img_1024', imgname + '.png')
    #                 path1 = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
    #                                     imgname + '.png')
    #                 maskpath = os.path.join('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024', imgname + '.png')
    #             # shutil.copy(path, args.output_dir)
    #             rgb_img = cv2.imread(path1, 1)[:, :, ::-1]
    #             rgb_img = cv2.resize(rgb_img, (512, 512)) / 255.
    #             rgb_imgs.append(rgb_img)
    #             img = Image.open(path)
    #             img = img.convert('RGB')
    #             maskimg = Image.open(maskpath)
    #             img, maskimg = transform(img, maskimg)
    #             imgs.append(img)
    #             img_names.append(imgname.split('.png')[0])
    #             del img, maskimg, rgb_img
    # # args.output_dir = args.output_dir + '/layer0EigenGradCAM/'
    #
    # ffs = [3]
    # for ee in range(1):
    #     for jj in range(len(ffs)):
    #         ff = ffs[jj]
    #         if ee == 0:
    #             args.output_dir = org_output_dir + '/GradCAM_guizhou' + args.experi + '/layers' + str(ff) + '/'
    #             print(args.output_dir)
    #             Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #             cam = GradCAM(model=model,
    #                           # target_layers=[model.fpn.smooth_convs],
    #                           # target_layers=[model.globalclassifier[0]],
    #                           target_layers=[model.layers[-1].blocks[-2]],
    #                           # target_layers=[model.stages[-1]],
    #                           # target_layers=[model.backbone.stages[-1]],
    #                           # target_layers=[model.blocks[-8].norm1],
    #                           # use_cuda=True,
    #                           reshape_transform=reshape_transform)
    #         for ii in range(len(imgs)):
    #             img = imgs[ii]
    #             rgb_img = rgb_imgs[ii]
    #             imgname = img_names[ii]
    #             print(ii, ': ', img_names[ii])
    #             # make the image divisible by the patch size
    #             w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    #             img = img[:, :w, :h].unsqueeze(0)
    #
    #             w_featmap = img.shape[-2] // args.patch_size
    #             h_featmap = img.shape[-1] // args.patch_size
    #             # output = model(img.to(device))
    #             output = model(img.to(device))
    #             ii+=1
    #             pred_name = 'Sick'
    #             grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
    #                 0)])
    #             # grayscale_cam0 = cam(input_tensor=img.to(device))
    #             grayscale_cam0 = grayscale_cam0[0, :]
    #             # 将 grad-cam 的输出叠加到原始图像上
    #             visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
    #             # 保存可视化结果
    #             cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    #             cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
    #                 0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
    #
    #             # aa = output.squeeze(0).argmax().item()
    #             # if aa == 1:
    #             #     pred_name = 'Health'
    #             #
    #             #     grayscale_cam1 = cam(input_tensor=img.to(device))
    #             #     grayscale_cam1 = grayscale_cam1[0, :]
    #             #     visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
    #             #     # 保存可视化结果
    #             #     cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    #             #     cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
    #             #         0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
    #             # else:
    #             #     pred_name = 'Sick'
    #             #     # grayscale_cam0 = cam(input_tensor=img.to(device), targets=[ClassifierOutputTarget(
    #             #     #     0)])
    #             #     grayscale_cam0 = cam(input_tensor=img.to(device))
    #             #     grayscale_cam0 = grayscale_cam0[0, :]
    #             #     # 将 grad-cam 的输出叠加到原始图像上
    #             #     visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
    #             #     # 保存可视化结果
    #             #     cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    #             #     cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
    #             #         0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)
