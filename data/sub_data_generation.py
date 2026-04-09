import random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import torchxrayvision as xrv
import skimage
import torchvision
def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img

import pandas as pd
from torchvision.transforms import functional as F
class Shanxi_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 sub_imgpath,
                 txtpath,
                 txt_sub_txt,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath=csvpath
        self.sub_img_size=sub_img_size
        self.csv = pd.read_excel(csvpath)
        self.sub_imgpath=sub_imgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio
        self.txt_sub_txt=txt_sub_txt
        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        train_file = open(self.txt_sub_txt, 'a')
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_imgs_final=[]
        imgs=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        self.img_size=image.size[0]

        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_labels.append(right_top_label)

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_labels.append(left_center_label)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*self.img_size
            ymin = values[0, 2]/1024*self.img_size
            xmax = values[0, 3]/1024*self.img_size
            ymax = values[0, 4]/1024*self.img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
        image = self.transforms(image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga = self.data_subregions_transform(subregions_imgs[ii])
                imgs.append(imga)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                imgs.append(imga)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, labels, imgname


from PIL import Image

class Shanxi_w7masks_Subregions_5classes_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 sub_imgpath,
                 txtpath,
                 txt_sub_txt,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath
        self.sub_path = sub_imgpath
        self.sub_imgpath=self.sub_path+'/subregionsimgs/'
        os.makedirs(os.path.dirname(self.sub_imgpath), exist_ok=True)
        self.sub_maskpath = self.sub_path + '/subregionsmasks/'
        os.makedirs(os.path.dirname(self.sub_maskpath), exist_ok=True)
        self.txt_sub_txt = txt_sub_txt

        self.sub_img_size=sub_img_size
        self.csv = pd.read_excel(csvpath)

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        txt_sub_file = open(self.txt_sub_txt, 'a')
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        subregions_labelss=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_size=1024
        # img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        # image = Image.open(img_path).convert('RGB')
        # img_size = image.size[0]
        # mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        # mask_image2 = Image.open(mask_img_path)
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        margin_y = 50  # 纵向上下各扩展 50 像素 (根据你的需求调整这个值)
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            # yinying_index=csv_line.columns.get_loc('阴影形态')
            yinying =csv_line['阴影形态'].iloc[0]
            print(imgname, yinying)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size

            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界

            height = ymax - ymin
            width = xmax - xmin
            left_top_label = values[0]
            # left_top_img = F.crop(image, ymin, xmin, height, width)
            # left_top_mask=F.crop(mask_image2, ymin, xmin, height, width)
            # left_top_img.save(self.sub_imgpath+imgname.split('.png')[0] + '_left_top.png')
            # left_top_mask.save(self.sub_maskpath+imgname.split('.png')[0] + '_left_top.png')
            # if left_top_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_top:' + left_top_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_top:'+left_top_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_left_top:' + left_top_label[0] + '\n')

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界

            height = ymax - ymin
            width = xmax - xmin
            right_top_label = values[0]
            # right_top_img = F.crop(image, ymin, xmin, height, width)
            # right_top_mask =F.crop(mask_image2, ymin, xmin, height, width)
            # right_top_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_top.png')
            # right_top_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_top.png')
            # if right_top_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_top:' + right_top_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_top:'+right_top_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_right_top:' + right_top_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            left_center_label = values[0]
            # left_center_img = F.crop(image, ymin, xmin, height, width)
            # left_center_mask = F.crop(mask_image2, ymin, xmin, height, width)
            # left_center_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_left_center.png')
            # left_center_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_left_center.png')
            # if left_center_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_center:' + left_center_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_center:'+left_center_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_left_center:' + left_center_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            right_center_label = values[0]
            # right_center_mask=F.crop(mask_image2, ymin, xmin, height, width)
            # right_center_img = F.crop(image, ymin, xmin, height, width)
            # right_center_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_center.png')
            # right_center_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_center.png')
            # if right_center_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_center:' + right_center_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_center:'+right_center_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_right_center:' + right_center_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_label = values[0]
            # left_bottom_img = F.crop(image, ymin, xmin, height, width)
            # left_bottom_mask=F.crop(mask_image2, ymin, xmin, height, width)
            # left_bottom_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_left_bottom.png')
            # left_bottom_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_left_bottom.png')
            # if left_bottom_label[0] != '0/0':
            #     txt_sub_file.write(
            #         imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + ',' + yinying + '\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_label = values[0]
            # right_bottom_img = F.crop(image, ymin, xmin, height, width)
            # right_bottom_mask=F.crop(mask_image2, ymin, xmin, height, width)
            # right_bottom_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_bottom.png')
            # right_bottom_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_bottom.png')
            # if right_bottom_label[0] != '0/0':
            #     txt_sub_file.write(
            #         imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + ',' + yinying + '\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + ',none\n')
            txt_sub_file.write(imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + '\n')

        return imgname
class Shanxi_w7masks_Subregions_5classes_Dataset2(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 sub_imgpath,
                 txtpath,
                 txt_sub_txt,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_Dataset2, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath
        self.sub_path = sub_imgpath
        self.sub_imgpath=self.sub_path+'/subregionsimgs/'
        os.makedirs(os.path.dirname(self.sub_imgpath), exist_ok=True)
        self.sub_maskpath = self.sub_path + '/subregionsmasks/'
        os.makedirs(os.path.dirname(self.sub_maskpath), exist_ok=True)
        self.txt_sub_txt = txt_sub_txt

        self.sub_img_size=sub_img_size
        self.csv = pd.read_excel(csvpath)

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        txt_sub_file = open(self.txt_sub_txt, 'a')
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        subregions_labelss=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_size=1024
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        margin_y = 50  # 纵向上下各扩展 50 像素 (根据你的需求调整这个值)
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            # yinying_index=csv_line.columns.get_loc('阴影形态')
            yinying =csv_line['阴影形态'].iloc[0]
            print(imgname, yinying)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size

            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界

            height = ymax - ymin
            width = xmax - xmin
            left_top_label = values[0]
            left_top_img = F.crop(image, ymin, xmin, height, width)
            left_top_mask=F.crop(mask_image2, ymin, xmin, height, width)
            left_top_img.save(self.sub_imgpath+imgname.split('.png')[0] + '_left_top.png')
            left_top_mask.save(self.sub_maskpath+imgname.split('.png')[0] + '_left_top.png')
            # if left_top_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_top:' + left_top_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_top:'+left_top_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_left_top:' + left_top_label[0] + '\n')

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界

            height = ymax - ymin
            width = xmax - xmin
            right_top_label = values[0]
            right_top_img = F.crop(image, ymin, xmin, height, width)
            right_top_mask =F.crop(mask_image2, ymin, xmin, height, width)
            right_top_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_top.png')
            right_top_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_top.png')
            # if right_top_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_top:' + right_top_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_top:'+right_top_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_right_top:' + right_top_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            left_center_label = values[0]
            left_center_img = F.crop(image, ymin, xmin, height, width)
            left_center_mask = F.crop(mask_image2, ymin, xmin, height, width)
            left_center_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_left_center.png')
            left_center_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_left_center.png')
            # if left_center_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_center:' + left_center_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_center:'+left_center_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_left_center:' + left_center_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            right_center_label = values[0]
            right_center_mask=F.crop(mask_image2, ymin, xmin, height, width)
            right_center_img = F.crop(image, ymin, xmin, height, width)
            right_center_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_center.png')
            right_center_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_center.png')
            # if right_center_label[0]!='0/0':
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_center:' + right_center_label[0] + ','+yinying+'\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_center:'+right_center_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_right_center:' + right_center_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_label = values[0]
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            left_bottom_mask=F.crop(mask_image2, ymin, xmin, height, width)
            left_bottom_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_left_bottom.png')
            left_bottom_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_left_bottom.png')
            # if left_bottom_label[0] != '0/0':
            #     txt_sub_file.write(
            #         imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + ',' + yinying + '\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_left_bottom:' + left_bottom_label[0] + '\n')


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            # --- 添加边距 ---
            # ymin = max(0, ymin - margin_y)  # 确保不超出图像上边界
            # ymax = min(img_size, ymax + margin_y)  # 确保不超出图像下边界
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_label = values[0]
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            right_bottom_mask=F.crop(mask_image2, ymin, xmin, height, width)
            right_bottom_img.save(self.sub_imgpath + imgname.split('.png')[0] + '_right_bottom.png')
            right_bottom_mask.save(self.sub_maskpath + imgname.split('.png')[0] + '_right_bottom.png')
            # if right_bottom_label[0] != '0/0':
            #     txt_sub_file.write(
            #         imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + ',' + yinying + '\n')
            # else:
            #     txt_sub_file.write(imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + ',none\n')
            # txt_sub_file.write(imgname.split('.png')[0] + '_right_bottom:' + right_bottom_label[0] + '\n')

        return imgname
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import transforms as pth_transform

    global_transfo2 = pth_transform.Compose([
        pth_transform.RandomResizedCrop(512, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        pth_transform.RandomHorizontalFlip(p=0.5),
        pth_transform.RandomRotation(degrees=(-15, 15)),
        pth_transform.RandomAutocontrast(p=0.3),
        pth_transform.RandomEqualize(p=0.3),
        # pth_transform.ToTensor(),
        # pth_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    global_transfo2_subregions = pth_transform.Compose([
        # pth_transform.Resize((512, 512),
        pth_transform.RandomResizedCrop(128, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        pth_transform.RandomHorizontalFlip(p=0.5),
        pth_transform.RandomRotation(degrees=(-15, 15)),
        pth_transform.RandomAutocontrast(p=0.3),
        pth_transform.RandomEqualize(p=0.3),
        # pth_transform.ToTensor(),
        # pth_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset=Shanxi_w7masks_Subregions_5classes_Dataset2('/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_img_1024',
                                              '/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_mask_1024',
                                              '/disk3/wjr/dataset/nejm/shanxidataset/',
                                              txtpath='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/trainval.txt',
                                              txt_sub_txt='//disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/temp.txt',
                                   csvpath='/disk3/wjr/dataset/nejm/shanxidataset/subregions_label_shanxi_all_woyinying.xlsx',
                                   data_transform=global_transfo2, data_subregions_transform=global_transfo2_subregions,
                                   sub_img_size=128,)
    dataset = Shanxi_w7masks_Subregions_5classes_Dataset2('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                                         '/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_mask_1024',
                                                         '/disk3/wjr/dataset/nejm/guizhoudataset/',
                                                         txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                                         txt_sub_txt='/disk3/wjr/dataset/nejm/guizhoudataset/temp.txt',
                                                         csvpath='/disk3/wjr/dataset/nejm/guizhoudataset/subregions_guizhou_all.xlsx',
                                                         data_transform=global_transfo2,
                                                         data_subregions_transform=global_transfo2_subregions,
                                                         sub_img_size=128, )
    data_loader = DataLoader(dataset,
                             batch_size=16,
                             shuffle=True,
                             pin_memory=True)

    for sample in data_loader:
        print(1)
