import os

import cv2
import numpy as np
import pandas as pd

import torch
import torchvision
from sklearn.utils import compute_class_weight
from PIL import Image
from torchvision.transforms import InterpolationMode
class ChestXRay14(torch.utils.data.Dataset):
    """PyTorch Dataset for NIH ChestXRay14 dataset."""

    def __init__(self, img_dir, txt_dir, csv_dir, data_transform=None, n_TTA=0, n=-1):
        self.img_dir = img_dir
        self.transform = data_transform
        self.n_TTA = n_TTA
        self.n = n
        self.csv_dir = csv_dir
        self.txt_dir=txt_dir
        # 读取 CSV 文件
        csv_data = pd.read_csv(self.csv_dir)
        self.CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax']
        # 创建一个字典以快速查找 Finding Labels
        finding_labels_dict = dict(zip(csv_data['Image Index'], csv_data['Finding Labels']))

        # 准备存储文件名和标签的列表
        file_names = []
        labels = []

        # 读取 test_list.txt 并填充文件名和标签
        with open(self.txt_dir, 'r') as test_file:
            for line in test_file:
                file_name = line.strip()  # 去掉空白字符
                file_names.append(file_name)

                # 获取对应的 Finding Labels
                if file_name in finding_labels_dict:
                    finding_labels = finding_labels_dict[file_name]
                    labels.append(finding_labels)
                else:
                    labels.append('No Finding Labels')  # 处理找不到的情况


        label_df = pd.DataFrame({'file': file_names, 'Finding Labels': labels})

        # 将 Finding Labels 列拆分为多列
        for pathology in self.CLASSES:
            label_df[pathology] = label_df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

        # 排序
        label_df = label_df.sort_values(by='file')

        # 提取文件名和标签
        self.files = label_df['file'].values.tolist()
        self.labels = label_df[self.CLASSES].values
        # 计算每个类别的类权重
        # self.class_weights = {}
        # for i in range(len(self.CLASSES)):
        #     # 提取每个类别的标签
        #     y_i = self.labels[:, i]
        #     # 计算类权重
        #     weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_i)
        #     self.class_weights[self.CLASSES[i]] = weight[1]  # 只取存在的类别的权重


        # Define augmentation pipeline for training and testing (when TTA enabled)
        if self.transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        # For validation and testing (with TTA disabled), simply resize to 224x224
        # else:
        #     self.transform = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         # torchvision.transforms.RandomCrop(224),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         # torchvision.transforms.RandomRotation(degrees=5),
        #         torchvision.transforms.ColorJitter(contrast=0.25),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])
        #     ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # x = cv2.imread(os.path.join(self.img_dir, self.files[idx]), cv2.IMREAD_GRAYSCALE)
        img_path = os.path.join(self.img_dir, self.files[idx])
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x)
        y = self.labels[idx]
        return x.float(), torch.from_numpy(y).float()