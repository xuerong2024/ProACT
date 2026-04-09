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
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import classification.util.misc as misc
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from classification.data import transforms
from torchvision import models as torchvision_models
import classification.utils
from classification.utils.logger import create_logger
from classification.utils.configv2 import get_config
from classification.models import build_model
from classification.utils_dino import DINOHead, MultiCropWrapper_mask, get_params_groups
from classification.util.datasets_pneum import Shanxi_Mask_Dataset_DINO as Shanxi_Mask_Dataset
from classification.util.datasets_pneum import Shanxi_Dataset_DINO as Shanxi_Dataset
from classification.utils.lr_scheduler import build_scheduler
from classification.utils.optimizer import build_optimizer
from classification.utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import sklearn, sklearn.metrics
import pandas as pd
import sklearn.metrics
import classification.utils_dino as utils_dino
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageOps, ImageEnhance
from scipy.ndimage.measurements import label
from scipy.ndimage import binary_dilation
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import random
torch.set_num_threads(3)
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
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    # Misc
    parser.add_argument('--data_path', default='/disk1/wjr/dataset/shanxi_dataset/', type=str,
        help='Please specify path to the ImageNet training data.')

    return parser

from torchvision import transforms as pth_transforms
def showimg(data_path):
    save_path='/disk1/wjr/dataset/shanxi_dataset/zaoshaiorg/visimg_dataaug3/'
    torch.manual_seed(42)
    flip_and_color_jitter = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=1
        ),
        # transforms.RandomGrayscale(p=0.2),
    ])
    # first global crop
    trans_color=transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
    # transfo_region = transforms.ThreeRegionsCenterCrop(96, interpolation=Image.BICUBIC)
    # transfo_region = transforms.RandomCenterResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC)
    transfo_region = utils_dino.random_bbox_maskimgs(0.8)
    global_transfo1 = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC),
        # flip_and_color_jitter,
        # utils_dino.GaussianBlur(1.0),
        utils_dino.random_bbox_maskimgs(1),
        # normalize,
    ])
    global_transfo2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        # utils_dino.GaussianBlur(0.1),
        utils_dino.Solarizationimgs(0.2),
        utils_dino.random_bbox_maskimgs(0.2),
        # normalize,
    ])
    random_views = transforms.Compose([
        # utils_dino.Texture_Enhance(p=0.1),
        # transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        # utils_dino.GaussianBlurMaskimgs(0.1),
        utils_dino.Solarizationimgs(0.2),
        utils_dino.random_bbox_maskimgs(0.2),
        # normalize,
    ])

    # transformation for the local small crops
    local_crops_number = 20
    # local_transfo_regions = transforms.ThreeRegionsCenterCrop(96, interpolation=Image.BICUBIC)
    local_transfo_regions = transforms.NineRegionsMixedMaskCenterCrop_GeJi(96, interpolation=Image.BICUBIC)

    local_transfo = transforms.Compose([
            transforms.RandomCenterResizedCropHW(96, scale=(0.8, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.random_bbox_maskimgs(0.5),
            # utils_dino.GaussianBlur(p=0.5),
        ])
    file_list = os.listdir(data_path)
    sick_filename='Sick_Stage1_geng-zai-yun_466904_M.png'
    health_filename='Health_wang-zheng_445045_M.png'
    # 构建完整的文件路径
    sick_file_path = os.path.join(data_path, sick_filename)
    # 读取图像
    sick_image = Image.open(sick_file_path).convert('RGB')
    health_file_path = os.path.join(data_path, health_filename)
    # 读取图像
    health_image = Image.open(health_file_path).convert('RGB')
    sick_mask_image = Image.open(os.path.join('/disk1/wjr/dataset/shanxi_dataset/segmask_ww_wl224', sick_filename))
    health_mask_image = Image.open(os.path.join('/disk1/wjr/dataset/shanxi_dataset/segmask_ww_wl224', sick_filename))

    image3, mask3 = local_transfo_regions(sick_image, health_image, sick_mask_image, health_mask_image, 185,192,180,181)

    for ii in range(len(image3)):
        local_image=image3[ii]
        # imga, maska = local_transfo(local_image, mask3[ii])
        plt.imshow(local_image)
        plt.show()
        # imga.save(os.path.join(save_path,
        #                                       sick_filename.split('.png')[0] + '_localimgtrans_' + str(ii) + '_' + str(
        #                                           ii) + '.png'))
            # # plt.imshow(image)
            # # plt.show()
            # for ii in range(20):
            #     # # global_image1,_ = transfo_region(image, mask_image)
            #     # global_image1, _=global_transfo1(image, mask_image)
            #     # # global_image2, _ = global_transfo2(image, mask_image)
            #     # global_image1.save(os.path.join(save_path, file_name.split('.png')[0] + '_globalimg1_' +str(ii) + '.png'))
            #     # # global_image2.save(os.path.join(save_path, file_name.split('.png')[0] + '_globalimg2_' +str(ii) + '.png'))
            #
            #     for jj in range(6):
            #         local_image, mask3 = local_transfo_regions(image, mask_image, jj)
            #         # plt.imshow(local_image)
            #         # plt.show()
            #         local_image, mask3 = local_transfo(local_image, mask3)
            #         # plt.imshow(local_image)
            #         # plt.show()
            #         # aa=np.array(local_image)
            #         local_image.save(os.path.join(save_path,
            #                                       file_name.split('.png')[0] + '_localimg_' + str(ii) + '_' + str(
            #                                           jj) + '.png'))




class flaten_region_mask(torch.nn.Module):
    def __init__(self, area_threshold=5, area_unit=30):
        super().__init__()
        self.area_threshold = area_threshold
        self.area_unit = area_unit

    def forward(self, img1, mask):
        image1_smooth = img1.filter(ImageFilter.GaussianBlur(radius=4))
        image1_variance = np.abs(np.array(img1) - np.array(image1_smooth))
        # 找出面积大于 4 的平缓区域
        # 找出方差小于 50 的区域,即为平缓区域
        labels, num_labels = label(image1_variance < self.area_threshold)
        # 计算每个连通区域的面积
        areas = [np.sum(labels == i) for i in range(1, num_labels + 1)]
        # 选出面积大于 4 的区域
        large_smooth_regions = []
        for i, area in enumerate(areas):
            if area > self.area_unit:
                large_smooth_regions.append(i + 1)
        # 将掩码应用到平缓区域
        result = np.array(img1)
        # for i, region in enumerate(large_smooth_regions):
        #     result[labels == region] = 0
        for region in large_smooth_regions:
            result[labels == region] = 0
            # region_mask = (labels == region)
            # region_boundary = binary_dilation(region_mask) ^ region_mask
            # result[region_boundary] = 0

        result[np.array(mask) == 0] = (np.array(img1))[np.array(mask) == 0]
        # 将结果转换回 Pillow Image 对象
        result_image = Image.fromarray(result)
        return result_image, mask

if __name__ == '__main__':
    import gc
    torch.set_num_threads(3)

    avg = showimg('/disk1/wjr/dataset/shanxi_dataset/zaoshaiorg/visimg_process/seg_rec_image_org/')
