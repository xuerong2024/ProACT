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
if __name__ == '__main__':
    csvpath2 = "/disk3/wjr/dataset/nejm/shanxidataset/subregions_label_shanxi_ilo_new.xlsx"
    csv2 = pd.read_excel(csvpath2)
    subregion_txtpath='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/val_sick.txt'
    png_path='/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_mask_img_1024/'
    savepath='/disk3/wjr/dataset/nejm/shanxidataset/cam_rec_lines/'
    Path(savepath).mkdir(parents=True, exist_ok=True)
    img_size=1024
    # 在图像上绘制矩形框和标签
    color = (0, 0, 255)  # BGR格式的红色
    thickness = 10  # 线宽
    with open(subregion_txtpath, 'r') as file:
        lines = file.readlines()
        for ii in range(len(lines)):
            line=lines[ii].strip()
            png_pathii = png_path + line + '.png'
            image = cv2.imread((png_pathii))
            csv_line = csv2.loc[(csv2["胸片名称"] == line)]
            if csv_line.size == 0:
                a = 1
            left_upper_index = csv_line.columns.get_loc('左上')
            # 获取 '左上' 列后三列的列索引
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            # 获取 '左上' 列后三列的值
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)



            left_upper_index = csv_line.columns.get_loc('右上')
            # 获取 '左上' 列后三列的列索引
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            # 获取 '左上' 列后三列的值
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            label = values[0, 0]
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            label = values[0, 0]
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            label = values[0, 0]
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            label = values[0, 0]
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            label = values[0, 0]
            # 绘制矩形框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            # gray_image = Image.fromarray(image, mode='L')  # 'L' 表示灰度模式

            # 保存图像
            cv2.imwrite(savepath+str(ii)+line + '.png', image)



