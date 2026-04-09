import os
import matplotlib
import glob
matplotlib.use('TkAgg')
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2
import xlrd
import numpy as np
from pathlib import Path
import tqdm
if __name__ == '__main__':
    mask_path='/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024/'
    png_path = '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024/'
    savepath='/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_img_1024/'
    Path(savepath).mkdir(parents=True, exist_ok=True)
    # 获取所有掩码文件
    mask_files = os.listdir(mask_path)
    for ii in range(len(mask_files)):
        mask_file=mask_files[ii]
        # 构建完整路径
        mask_full_path = os.path.join(mask_path, mask_file)
        png_full_path = os.path.join(png_path, mask_file)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        # 读取原始PNG图像（彩色或灰度）
        png = cv2.imread(png_full_path, cv2.IMREAD_GRAYSCALE)
        # 将掩码转换为浮点数并归一化到[0,1]范围
        mask_float = mask.astype(np.float32) / 255.0
        result = (png.astype(np.float32) * mask_float).astype(np.uint8)

        # 保存结果
        save_file = os.path.join(savepath, mask_file)
        cv2.imwrite(save_file, result)
    print("所有图像处理完成！")

