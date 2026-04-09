# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    if isinstance(tensor, tuple):
        tensor=tensor[0]
    # tensor=torch.cat((tensor[:, :98, :],tensor[:, 99:, :]),dim=1)
    # # tensor = tensor[:, 1:, :]
    # result = tensor.reshape(tensor.size(0),
    # height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    # result = tensor.permute(0,3,1,2)
    return tensor
if __name__ == '__main__':
    import shutil
    import glob
    from classification.utils.configv2 import get_config
    from classification.models import build_model
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pathlib import Path

    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--cfg', type=str,
                        default='/disk1/wjr/workspace/sec_two/VMamba-main-5-4090/classification/configs/vmamba/vmambav2v_tiny_224.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # parser.add_argument('--pretrained', type=str,
    #                     help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--pretrained_weights', type=str,
                        default='/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_org/contra_experi/vision_mamba/vision_mamba_wwwl224/vision_mamba_tiny_4_lr8e-05/bestpt.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    # parser.add_argument('--pretrained_weights', type=str,
    #                     default='/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_org/vmamba2new/vmambav2new_cross_sequence/vssm1_tiny_0230s_4_lr0.0001/bestpt.pth',
    #                     help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    # for acceleration
    # parser.add_argument('--fused_window_process', action='store_true',
    # help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb

    # EMA related parameters

    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument("--image_path", default='/disk1/wjr/dataset/shanxi_dataset/segimg_ww_wl224/', type=str,
                        help="Path of the image to load.")
    parser.add_argument("--txtpath", default='/disk1/wjr/dataset/shanxi_dataset/zaoshaiorg/val', type=str,
                        help="Path of the image to load.")

    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    # parser.add_argument('--output_dir',
    #                     default='/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_org/vmamba2new/vmambav2new_cross_sequence/',
    #                     help='Path where to save visualizations.')
    parser.add_argument('--output_dir',
                        default='/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_org/vmamba2new/our_localtop1_new/0num0.1self_loss1cls_weight0.5cls_local_weight_7550',
                        help='Path where to save visualizations.')
    # parser.add_argument('--output_dir',
    #                     default='/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_org/vmamba2new/tab2-LGVM-woSHFS/1num0.1self_loss1cls_weight0.5cls_local_weight/',
    #                     help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.5, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()
    args.experi='1'
    args.base_lr=1e-4
    args.patch_size=16
    args.txtpath = '/disk1/wjr/dataset/shanxi_dataset/jingbiao.txt'
    # args.output_dir = args.output_dir + '/vssm1_tiny_0230s_' + args.experi + '_lr' + str(
    #     args.base_lr)
    # args.pretrained_weights=args.output_dir + '/bestpt.pth'
    args.output_dir = '/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/9mixedregions_our_score-based/s-vmamba/0seed_teacher_ema0.992_head1024_freezeepoch5_1cls_weight0.5self_loss1cls_local_weight/lr_3e-05'
    args.pretrained_weights = args.output_dir + '/checkpoint_best.pth.tar'
    # args.pretrained_weights = args.output_dir + '/model_best.pth'
    config = get_config(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = build_model(config)
    model.classifier.head = nn.Linear(768, 2)
    model.cuda()

    # for p in model.parameters():
    #     p.requires_grad = False
    # model.eval()
    model.to(device)

    # state_dict = torch.load(args.pretrained_weights, map_location="cpu")['model']
    # model.load_state_dict(state_dict, strict=False)

    if os.path.isfile(args.pretrained_weights):
        # state_dict = torch.load(args.pretrained_weights, map_location="cpu")['model']
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)

        # # state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        # state_dict = torch.load(args.pretrained_weights, map_location="cpu")['student']
        # if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        #     print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        #     state_dict = state_dict[args.checkpoint_key]
        # # remove `module.` prefix
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # aa=model.state_dict()
        # msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    for name, param in model.named_parameters():
        print(name)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # open image
    if args.image_path is None:
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
        with open(args.txtpath, 'r', encoding='gbk') as file:
            lines = file.readlines()
        for line in lines:
            imgname = line.split('\n')[0]
            labelname = imgname.split('_')[0]
            if '.png' in imgname:
                path = os.path.join(args.image_path, imgname)
            else:
                path = os.path.join(args.image_path, imgname + '.png')

            # shutil.copy(path, args.output_dir)
            rgb_img = cv2.imread(path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224)) / 255.
            rgb_imgs.append(rgb_img)
            img = Image.open(path)
            img = img.convert('RGB')
            img = transform(img)
            imgs.append(img)
            img_names.append(imgname.split('.png')[0])
    # args.output_dir = args.output_dir + '/layer0EigenGradCAM/'
    org_output_dir = args.output_dir
    ffs = [0, 1, 2, 3]
    for ee in range(1):
        for jj in range(1):
            ff = ffs[jj]
            if ee == 0:
                args.output_dir = org_output_dir + '/GradCAM/layers' + str(ff) + '/'
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                cam = GradCAM(model=model,
                              target_layers=[model.layers[-1].blocks[-2]],
                              # target_layers=[model.blocks[-8].norm1],
                              # use_cuda=True,
                              reshape_transform=reshape_transform)
            # elif ee == 1:
            #     args.output_dir = org_output_dir + '/XGradCAM/layers' + str(ff) + '/'
            #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            #     cam = XGradCAM(model=model,
            #                    target_layers=[model.layers[-2].mixer],
            #                    # target_layers=[model.blocks[-8].norm1],
            #                    # use_cuda=True,
            #                    reshape_transform=reshape_transform)
            # elif ee == 2:
            #     args.output_dir = org_output_dir + '/EigenGradCAM/layers' + str(ff) + '/'
            #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            #     cam = EigenGradCAM(model=model,
            #                        target_layers=[model.layers[-2].mixer],
            #                        # target_layers=[model.blocks[-8].norm1],
            #                        # use_cuda=True,
            #                        reshape_transform=reshape_transform)
            for ii in range(len(imgs)):
                img = imgs[ii]
                rgbimg = rgb_imgs[ii]
                imgname = img_names[ii]
                # make the image divisible by the patch size
                w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
                img = img[:, :w, :h].unsqueeze(0)

                w_featmap = img.shape[-2] // args.patch_size
                h_featmap = img.shape[-1] // args.patch_size
                output = model(img.to(device))
                # attentions = model.get_last_selfattention(img.to(device))

                # grayscale_cam = cam(input_tensor=img.to(device), targets=None)
                # aa = output.squeeze(0).argmax().item()
                # if aa == 1:
                #     pred_name = 'Health'
                # else:
                #     pred_name = 'Sick'
                # grayscale_cam = grayscale_cam[0, :]
                # # 将 grad-cam 的输出叠加到原始图像上
                # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                # # 保存可视化结果
                # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                # cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                #     0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)

                grayscale_cam0 = cam(input_tensor=img.to(device), targets = [ClassifierOutputTarget(
                        0)])
                grayscale_cam1 = cam(input_tensor=img.to(device), targets = [ClassifierOutputTarget(
                        1)])
                pred_name = 'Health'
                grayscale_cam1 = grayscale_cam1[0, :]
                # 将 grad-cam 的输出叠加到原始图像上
                visualization = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
                # 保存可视化结果
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                    0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)

                pred_name = 'Sick'
                grayscale_cam0 = grayscale_cam0[0, :]
                # 将 grad-cam 的输出叠加到原始图像上
                visualization = show_cam_on_image(rgb_img, grayscale_cam0, use_rgb=True)
                # 保存可视化结果
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
                cv2.imwrite(os.path.join(args.output_dir, str(ii) + imgname.split('.png')[0].split('_')[
                    0] + '_pred_' + pred_name + '_map' + imgname.split('.png')[0] + '.png'), visualization)




