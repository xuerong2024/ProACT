import random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from pathlib import Path
import numpy as np
import torch

class Guiyang_Feiqu_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 ):
        super(Guiyang_Feiqu_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.txtpath = txtpath
        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        self.line_0_0=[]
        self.line_0_1 = []
        self.line_1_0 = []
        self.line_1_1 = []

        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        for line in self.lines:
            if '0_0' in line:
                self.line_0_0.append(line)
            elif '0_1' in line:
                self.line_0_1.append(line)
            elif '1_0' in line:
                self.line_1_0.append(line)
            else:
                self.line_1_1.append(line)

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[-3]
        labelname = imgname.split(labelname)[-1]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('0_0' in labelname)
        label.append('0_1' in labelname)
        label.append('1_0' in labelname)
        label.append('1_1' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        return image, label, imgname

class Fibrosis_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        # a=1

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname
class Fibrosis_wmask_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_wmask_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        # a=1

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_img_path = os.path.join(self.maskpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path)

        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # sample["lab"] = label
        # sample["img"] = image
        # sample["img_name"]=imgname
        return image, mask_image, label, imgname
class ILO_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(ILO_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.jpg')[0] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname
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
class Subregion_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskpath,
                 txtpath,
                 data_transform=None,
                 inference=True
                 ):
        super(Subregion_Dataset, self).__init__()
        self.imgpath = imgpath
        self.maskpath=maskpath
        self.inference=inference
        self.txtpath = txtpath
        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform


        with open(txtpath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname = line.split(':')[0]
        labelname = line.split(':')[-1]
        # print('labelname', labelname)
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        mask_path = os.path.join(self.maskpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        image, mask_image = self.transforms(image, mask_image)
        if '0/0' in labelname:
            label=4
        elif '0/1' in labelname:
            label=3
        elif '1/0' in labelname:
            label=2
        elif '1/1' in labelname:
            label=1
        else:
            label=0
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # print('label', label)

        return image, mask_image, label, imgname

class Shanxi_XRV_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_XRV_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')

        img = Image.open(img_path)
        # img = skimage.io.imread(img_path)
        img = self.transforms(img)
        # img = torch.from_numpy(img)
        img = (2 * img - 1.) * 1024
        # aa=torch.min(img)
        # img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        # img = img.mean(2)[None, ...]  # Make single color channel
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        sample["lab"] = label
        sample["img"] = img
        sample["img_name"]=imgname
        return img, label, imgname
class Fibrosis_XRV_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_XRV_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        # a=1

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path)
        image = self.transforms(image)
        image = (2 * image - 1.) * 1024

        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname
class Shanxi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname

class Shanxi_nih_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 nihpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_nih_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.nihpath=nihpath

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx].strip()
        imgname=line.split('.png ')[0]
        labelname = line.split('.png ')[-1]
        if 'Sick' in imgname or 'Health' in imgname:
            img_path=os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        else:
            img_path=os.path.join(self.nihpath, imgname.split('.png')[0] + '.png')

        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('No Finding' not in labelname)
        label.append('No Finding' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname
import pandas as pd
from torchvision.transforms import functional as F
class Shanxi_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
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

class Shanxi_wmasks_AllSubregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmasks_AllSubregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.csvpath=csvpath
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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        imgs=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        self.img_size=image.size[0]
        mask_image = Image.open(mask_img_path)
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
        else:
            w = int(round(self.img_size / 2))
            h = int(round(5/6*self.img_size / 3))

            xmin = 0
            ymin = 0
            xmax = w
            ymax = h
            left_top_label = np.array([-1., -1.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_labels.append(left_top_label)

            xmin = w
            ymin = 0
            xmax = self.img_size
            ymax = h
            right_top_label = np.array([-1., -1.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_labels.append(right_top_label)

            xmin = 0
            ymin = h
            xmax = w
            ymax = 2*h
            left_center_label = np.array([-1., -1.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_labels.append(left_center_label)

            xmin = w
            ymin = h
            xmax = self.img_size
            ymax = 2*h
            right_center_label = np.array([-1., -1.])
            height = ymax - ymin
            width = xmax - xmin
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            xmin = 0
            ymin = h
            xmax = w
            ymax =  self.img_size
            left_bottom_label = np.array([-1., -1.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            xmin = w
            ymin = 2 * h
            xmax = self.img_size
            ymax = self.img_size
            right_bottom_label = np.array([-1., -1.])
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
        for ii in range(len(subregions_imgs)):
            # image, mask = self.local_transfo_mask(image, mask)
            imga = self.data_subregions_transform(subregions_imgs[ii])
            imgs.append(imga)
            labels.append(subregions_labels[ii])
        return imgs, labels, imgname
from PIL import Image
import matplotlib.pyplot as plt
class Shanxi_wmask_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname
class Shanxi_wmask_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path)


        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_image)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, masks, labels, imgname
class Shanxi_w2masks_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 maskimgpath2,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w2masks_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.maskimgpath2 = maskimgpath2
        self.csvpath=csvpath

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
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_img_path2 = os.path.join(self.maskimgpath2, imgname.split('.png')[0] + '.png')
        if os.path.exists(mask_img_path2) and os.path.isfile(mask_img_path2):
            mask_image = Image.open(mask_img_path2)
            mask_image2 = Image.open(mask_img_path)
            masks_index.append(2)
        else:
            mask_image = Image.open(mask_img_path)
            mask_image2 = Image.open(mask_img_path)
            masks_index.append(1)


        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_image)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, masks, labels, masks_index, imgname
class Shanxi_w7masks_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)


        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[int(ymin):int(ymax), int(xmin):int(xmax)]



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin

            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin

            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin

            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin

            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        return imgs, masks, labels, masks_index, imgname
class Shanxi_w7masks_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)


        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[int(ymin):int(ymax), int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, masks, labels, masks_index, imgname
class Shanxi_w7masks_Subregions_5classes_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
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
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)


        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[int(ymin):int(ymax), int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                subregions_labelss.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                subregions_labelss.append(np.array([-1]))
            # aa=2

        return imgs, masks, labels, subregions_labelss, masks_index, imgname
class Shanxi_w7masks_Subregions_Mixed_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_Mixed_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label = np.array([0., 1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label = np.array([0., 1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label = np.array([0., 1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label = np.array([0., 1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label = np.array([0., 1.])
            else:
                left_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label = np.array([0., 1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    if 'Health' in line2:
                        choosen_index = 1
                        img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                        image2 = Image.open(img_path2).convert('RGB')
                        img_size = image2.size[0]
                        mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                        mask_image2_2 = Image.open(mask_img_path2)
                        mask_new_subimage_org2 = np.asarray(mask_image2_2)
                        mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                        left_upper_index2 = csv_line2.columns.get_loc('左上')
                        next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                        values = csv_line2[next_three_columns2].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_top_label2 = values[0, 0]
                        if '0/0' in left_top_label2 or '0/1' in left_top_label2:
                            left_top_label2 = np.array([0., 1.])
                        else:
                            left_top_label2 = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                        mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                               int(ymin):int(ymax),
                                                                                               int(xmin):int(xmax)]
                        subregions_imgs2.append(left_top_img2)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(left_top_label2)

                        left_upper_index = csv_line2.columns.get_loc('右上')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_top_label = values[0, 0]
                        if '0/0' in right_top_label or '0/1' in right_top_label:
                            right_top_label = np.array([0., 1.])
                        else:
                            right_top_label = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        right_top_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(right_top_img)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(right_top_label)
                        mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                                int(ymin):int(ymax),
                                                                                                int(xmin):int(xmax)]

                        # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                        # plt.show()
                        left_upper_index = csv_line2.columns.get_loc('左中')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_center_label = values[0, 0]
                        if '0/0' in left_center_label or '0/1' in left_center_label:
                            left_center_label = np.array([0., 1.])
                        else:
                            left_center_label = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_center_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(left_center_img)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(left_center_label)
                        mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('右中')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_center_label = values[0, 0]
                        if '0/0' in right_center_label or '0/1' in right_center_label:
                            right_center_label = np.array([0., 1.])
                        else:
                            right_center_label = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        right_center_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(right_center_img)
                        subregions_labels2.append(right_center_label)
                        mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('左下')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_bottom_label = values[0, 0]
                        if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                            left_bottom_label = np.array([0., 1.])
                        else:
                            left_bottom_label = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_imgs2.append(left_bottom_img)
                        subregions_labels2.append(left_bottom_label)
                        mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('右下')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_bottom_label = values[0, 0]
                        if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                            right_bottom_label = np.array([0., 1.])
                        else:
                            right_bottom_label = np.array([1., 0.])
                        height = ymax - ymin
                        width = xmax - xmin
                        right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_imgs2.append(right_bottom_img)
                        subregions_labels2.append(right_bottom_label)
                        mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]
                        image2_np = np.asarray(image2)
                        image_np = np.asarray(image)
                        mixed_img1 = np.expand_dims((
                                                                mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2)//255,
                                                    axis=-1) * image2_np + np.expand_dims(
                            (mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop)//255,
                            axis=-1)* image_np
                        mixed_img2 = np.expand_dims(
                            (
                                        mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop)//255,
                            axis=-1) * image_np + np.expand_dims(
                            (
                                        mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2)//255,
                            axis=-1) * image2_np

                        mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                        mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')


                        mixed_img1 = Image.fromarray(mixed_img1)
                        mixed_img2 = Image.fromarray(mixed_img2)
                        # plt.imshow(image)
                        # plt.show()
                        # plt.imshow(image2)
                        # plt.show()
                        # plt.imshow(mixed_img1)
                        # plt.show()
                        # plt.imshow(mixed_img2)
                        # plt.show()
                        mixed_label1 = subregions_labels2[1] + subregions_labels2[3] + subregions_labels2[5] + \
                                       subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                        mixed_label2 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5] + \
                                       subregions_labels2[0] + subregions_labels2[2] + subregions_labels2[4]
                        if mixed_label1[0] >= 2:
                            mixed_label1 = np.array([1., 0.])
                        else:
                            mixed_label1 = np.array([0., 1.])
                        if mixed_label2[0] >= 2:
                            mixed_label2 = np.array([1., 0.])
                        else:
                            mixed_label2 = np.array([0., 1.])


        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_imgs!=[]:
            mixed_img1, mixed_mask1 = self.transforms(mixed_img1, mixed_mask1)
            mixed_img2, mixed_mask2 = self.transforms(mixed_img2, mixed_mask2)
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))


        return imgs, masks, labels, masks_index, imgname
class Shanxi_w7masks_Subregions_Mixednew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_Mixednew_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        subregions_labels_mixed = []
        subregions_labels_mixed2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label = np.array([0., 1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label = np.array([0., 1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label = np.array([0., 1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label = np.array([0., 1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label = np.array([0., 1.])
            else:
                left_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label = np.array([0., 1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('右上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0, 0]
                    if '0/0' in left_top_label2 or '0/1' in left_top_label2:
                        left_top_label2 = np.array([0., 1.])
                    else:
                        left_top_label2 = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('左上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0, 0]
                    if '0/0' in right_top_label or '0/1' in right_top_label:
                        right_top_label = np.array([0., 1.])
                    else:
                        right_top_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0, 0]
                    if '0/0' in left_center_label or '0/1' in left_center_label:
                        left_center_label = np.array([0., 1.])
                    else:
                        left_center_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0, 0]
                    if '0/0' in right_center_label or '0/1' in right_center_label:
                        right_center_label = np.array([0., 1.])
                    else:
                        right_center_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0, 0]
                    if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                        left_bottom_label = np.array([0., 1.])
                    else:
                        left_bottom_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0, 0]
                    if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                        right_bottom_label = np.array([0., 1.])
                    else:
                        right_bottom_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)
                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    mixed_label1 = subregions_labels2[1] + subregions_labels2[3] + subregions_labels2[5] + \
                                   subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                    mixed_label2 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5] + \
                                   subregions_labels2[0] + subregions_labels2[2] + subregions_labels2[4]
                    if mixed_label1[0] >= 2:
                        mixed_label1 = np.array([1., 0.])
                    else:
                        mixed_label1 = np.array([0., 1.])
                    if mixed_label2[0] >= 2:
                        mixed_label2 = np.array([1., 0.])
                    else:
                        mixed_label2 = np.array([0., 1.])
                else:
                    if 'Health' in line2:
                        choosen_index = 1
                        img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                        image2 = Image.open(img_path2).convert('RGB')
                        img_size = image2.size[0]
                        mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                        mask_image2_2 = Image.open(mask_img_path2)
                        mask_new_subimage_org2 = np.asarray(mask_image2_2)
                        mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_left_subimage_org2[:, mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                                                                                                :,
                                                                                                mask_new_subimage_org2.shape[
                                                                                                    1] // 2:]
                        mask_new_right_subimage_org2[:, :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                                                                                                 :, :
                                                                                                    mask_new_subimage_org2.shape[
                                                                                                        1] // 2]

                        org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                        org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                        image2_np = np.asarray(image2)
                        image_np = np.asarray(image)

                        mixed_img1 = np.expand_dims(
                            mask_new_left_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(
                            org_right_mask // 255,
                            axis=-1) * image_np
                        mixed_img2 = np.expand_dims(
                            mask_new_right_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                                                                  axis=-1) * image_np
                        mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                        mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                        mixed_img1 = Image.fromarray(mixed_img1)
                        mixed_img2 = Image.fromarray(mixed_img2)

                        mixed_label2 = subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                        mixed_label1 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5]
                        subregions_labels2.append(np.array([0., 1.]))
                        subregions_labels2.append(np.array([0., 1.]))
                        subregions_labels2.append(np.array([0., 1.]))
                        subregions_labels2.append(np.array([0., 1.]))
                        subregions_labels2.append(np.array([0., 1.]))
                        subregions_labels2.append(np.array([0., 1.]))

                        if mixed_label1[0] >= 2:
                            mixed_label1 = np.array([1., 0.])
                        else:
                            mixed_label1 = np.array([0., 1.])
                        if mixed_label2[0] >= 2:
                            mixed_label2 = np.array([1., 0.])
                        else:
                            mixed_label2 = np.array([0., 1.])

        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)

        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])



            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])

            mixed_img1, mixed_mask1s = self.transforms(mixed_img1, mixed_mask1s)
            mixed_img2, mixed_mask2s = self.transforms(mixed_img2, mixed_mask2s)
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask1s[0])
            # plt.show()
            # plt.imshow(mixed_mask1s[-1])
            # plt.show()
            # plt.imshow(mixed_mask1s[-2])
            # plt.show()
            # plt.imshow(mixed_img2)
            # plt.show()
            # plt.imshow(mixed_mask2s[0])
            # plt.show()
            # plt.imshow(mixed_mask2s[-1])
            # plt.show()
            # plt.imshow(mixed_mask2s[-2])
            # plt.show()
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1s)
            masks.append(mixed_mask1s)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                # labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                subregions_labels.append(np.array([-1,-1]))
                mixed_sublabels.append(np.array([-1, -1]))
                mixed_sublabels2.append(np.array([-1,-1]))
                # labels.append(np.array([-1,-1]))
        return imgs, masks, labels, subregions_labels, mixed_sublabels, mixed_sublabels2, masks_index, imgname
class Shanxi_w7masks_Subregions_MixedSickHealth_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_MixedSickHealth_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label = np.array([0., 1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label = np.array([0., 1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label = np.array([0., 1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label = np.array([0., 1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label = np.array([0., 1.])
            else:
                left_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label = np.array([0., 1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('右上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0, 0]
                    if '0/0' in left_top_label2 or '0/1' in left_top_label2:
                        left_top_label2 = np.array([0., 1.])
                    else:
                        left_top_label2 = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('左上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0, 0]
                    if '0/0' in right_top_label or '0/1' in right_top_label:
                        right_top_label = np.array([0., 1.])
                    else:
                        right_top_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0, 0]
                    if '0/0' in left_center_label or '0/1' in left_center_label:
                        left_center_label = np.array([0., 1.])
                    else:
                        left_center_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0, 0]
                    if '0/0' in right_center_label or '0/1' in right_center_label:
                        right_center_label = np.array([0., 1.])
                    else:
                        right_center_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0, 0]
                    if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                        left_bottom_label = np.array([0., 1.])
                    else:
                        left_bottom_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0, 0]
                    if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                        right_bottom_label = np.array([0., 1.])
                    else:
                        right_bottom_label = np.array([1., 0.])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)
                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    mixed_label1 = subregions_labels2[1] + subregions_labels2[3] + subregions_labels2[5] + \
                                   subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                    mixed_label2 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5] + \
                                   subregions_labels2[0] + subregions_labels2[2] + subregions_labels2[4]
                    if mixed_label1[0] >= 2:
                        mixed_label1 = np.array([1., 0.])
                    else:
                        mixed_label1 = np.array([0., 1.])
                    if mixed_label2[0] >= 2:
                        mixed_label2 = np.array([1., 0.])
                    else:
                        mixed_label2 = np.array([0., 1.])
                else:
                    if 'Health' in line2:
                        choosen_index = 1
                        img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                        image2 = Image.open(img_path2).convert('RGB')
                        img_size = image2.size[0]
                        mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                        mask_image2_2 = Image.open(mask_img_path2)
                        mask_new_subimage_org2 = np.asarray(mask_image2_2)
                        mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_left_subimage_org2[:, mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                                                                                                :,
                                                                                                mask_new_subimage_org2.shape[
                                                                                                    1] // 2:]
                        mask_new_right_subimage_org2[:, :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                                                                                                 :, :
                                                                                                    mask_new_subimage_org2.shape[
                                                                                                        1] // 2]

                        org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                        org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                        image2_np = np.asarray(image2)
                        image_np = np.asarray(image)

                        mixed_img1 = np.expand_dims(
                            mask_new_left_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(
                            org_right_mask // 255,
                            axis=-1) * image_np
                        mixed_img2 = np.expand_dims(
                            mask_new_right_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                                                                  axis=-1) * image_np
                        mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                        mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                        mixed_img1 = Image.fromarray(mixed_img1)
                        mixed_img2 = Image.fromarray(mixed_img2)

                        mixed_label2 = subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                        mixed_label1 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5]

                        if mixed_label1[0] >= 2:
                            mixed_label1 = np.array([1., 0.])
                        else:
                            mixed_label1 = np.array([0., 1.])
                        if mixed_label2[0] >= 2:
                            mixed_label2 = np.array([1., 0.])
                        else:
                            mixed_label2 = np.array([0., 1.])




        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_imgs!=[]:
            mixed_img1, mixed_mask1 = self.transforms(mixed_img1, mixed_mask1)
            mixed_img2, mixed_mask2 = self.transforms(mixed_img2, mixed_mask2)
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))
        return imgs, masks, labels, masks_index, imgname
class Shanxi_w7masks_Subregions_5classes_MixedSickHealth_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_MixedSickHealth_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])


            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])


            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('左上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0]
                    if '0/0' in left_top_label2:
                        left_top_label2 = np.array([4])
                    elif '0/1' in left_top_label2:
                        left_top_label2 = np.array([3])
                    elif '1/0' in left_top_label2:
                        left_top_label2 = np.array([2])
                    elif '1/1' in left_top_label2:
                        left_top_label2 = np.array([1])
                    else:
                        left_top_label2 = np.array([0])

                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('右上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0]
                    if '0/0' in right_top_label:
                        right_top_label = np.array([4])
                    elif '0/1' in right_top_label:
                        right_top_label = np.array([3])
                    elif '1/0' in right_top_label:
                        right_top_label = np.array([2])
                    elif '1/1' in right_top_label:
                        right_top_label = np.array([1])
                    else:
                        right_top_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0]
                    if '0/0' in left_center_label:
                        left_center_label = np.array([4])
                    elif '0/1' in left_center_label:
                        left_center_label = np.array([3])
                    elif '1/0' in left_center_label:
                        left_center_label = np.array([2])
                    elif '1/1' in left_center_label:
                        left_center_label = np.array([1])
                    else:
                        left_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0]
                    if '0/0' in right_center_label:
                        right_center_label = np.array([4])
                    elif '0/1' in right_center_label:
                        right_center_label = np.array([3])
                    elif '1/0' in right_center_label:
                        right_center_label = np.array([2])
                    elif '1/1' in right_center_label:
                        right_center_label = np.array([1])
                    else:
                        right_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0]
                    if '0/0' in left_bottom_label:
                        left_bottom_label = np.array([4])
                    elif '0/1' in left_bottom_label:
                        left_bottom_label = np.array([3])
                    elif '1/0' in left_bottom_label:
                        left_bottom_label = np.array([2])
                    elif '1/1' in left_bottom_label:
                        left_bottom_label = np.array([1])
                    else:
                        left_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0]
                    if '0/0' in right_bottom_label:
                        right_bottom_label = np.array([4])
                    elif '0/1' in right_bottom_label:
                        right_bottom_label = np.array([3])
                    elif '1/0' in right_bottom_label:
                        right_bottom_label = np.array([2])
                    elif '1/1' in right_bottom_label:
                        right_bottom_label = np.array([1])
                    else:
                        right_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)
                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(1 for x in values1 if x == 4 or x == 3)
                    if count1 <= 4:
                        mixed_label1 = np.array([1., 0.])
                    else:
                        mixed_label1 = np.array([0., 1.])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(1 for x in values2 if x == 4 or x == 3)
                    if count2 <= 4:
                        mixed_label2 = np.array([1., 0.])
                    else:
                        mixed_label2 = np.array([0., 1.])

                    # mixed_label1 = subregions_labels2[1] + subregions_labels2[3] + subregions_labels2[5] + \
                    #                subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                    # mixed_label2 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5] + \
                    #                subregions_labels2[0] + subregions_labels2[2] + subregions_labels2[4]
                    # if mixed_label1[0] >= 2:
                    #     mixed_label1 = np.array([1., 0.])
                    # else:
                    #     mixed_label1 = np.array([0., 1.])
                    # if mixed_label2[0] >= 2:
                    #     mixed_label2 = np.array([1., 0.])
                    # else:
                    #     mixed_label2 = np.array([0., 1.])
                else:
                    if 'Health' in line2:
                        choosen_index = 1
                        img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                        image2 = Image.open(img_path2).convert('RGB')
                        img_size = image2.size[0]
                        mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                        mask_image2_2 = Image.open(mask_img_path2)
                        mask_new_subimage_org2 = np.asarray(mask_image2_2)
                        mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_left_subimage_org2[:, mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                                                                                                :,
                                                                                                mask_new_subimage_org2.shape[
                                                                                                    1] // 2:]
                        mask_new_right_subimage_org2[:, :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                                                                                                 :, :
                                                                                                    mask_new_subimage_org2.shape[
                                                                                                        1] // 2]

                        org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                        org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                        image2_np = np.asarray(image2)
                        image_np = np.asarray(image)

                        mixed_img1 = np.expand_dims(
                            mask_new_left_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(
                            org_right_mask // 255,
                            axis=-1) * image_np
                        mixed_img2 = np.expand_dims(
                            mask_new_right_subimage_org2 // 255,
                            axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                                                                  axis=-1) * image_np
                        mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                        mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                        mixed_img1 = Image.fromarray(mixed_img1)
                        mixed_img2 = Image.fromarray(mixed_img2)

                        values1 = [
                            subregions_labels[0],  # 0
                            subregions_labels[2],  # 1
                            subregions_labels[4],  # 0
                        ]
                        count1 = sum(1 for x in values1 if x == 4 or x == 3)
                        if count1 <= 1:
                            mixed_label2 = np.array([1., 0.])
                        else:
                            mixed_label2 = np.array([0., 1.])

                        values2 = [
                            subregions_labels[1],  # 0
                            subregions_labels[3],  # 1
                            subregions_labels[5],  # 0
                        ]
                        count2 = sum(1 for x in values2 if x == 4 or x == 3)
                        if count2 <= 1:
                            mixed_label1 = np.array([1., 0.])
                        else:
                            mixed_label1 = np.array([0., 1.])


                        # mixed_label2 = subregions_labels[0] + subregions_labels[2] + subregions_labels[4]
                        # mixed_label1 = subregions_labels[1] + subregions_labels[3] + subregions_labels[5]
                        # if mixed_label1[0] >= 2:
                        #     mixed_label1 = np.array([1., 0.])
                        # else:
                        #     mixed_label1 = np.array([0., 1.])
                        # if mixed_label2[0] >= 2:
                        #     mixed_label2 = np.array([1., 0.])
                        # else:
                        #     mixed_label2 = np.array([0., 1.])




        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        sub_labels = []
        if subregions_imgs!=[]:
            mixed_img1, mixed_mask1 = self.transforms(mixed_img1, mixed_mask1)
            mixed_img2, mixed_mask2 = self.transforms(mixed_img2, mixed_mask2)
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                sub_labels.append(subregions_labels[ii])
                # labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                sub_labels.append(np.array([-1]))
                # labels.append(np.array([-1,-1]))
        return imgs, masks, labels, sub_labels, masks_index, imgname
class Shanxi_w7masks_Subregions_5classes_Mixednew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_Mixednew_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        subregions_labels_mixed = []
        subregions_labels_mixed2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('右上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0]
                    if '0/0' in left_top_label2:
                        left_top_label2 = np.array([4])
                    elif '0/1' in left_top_label2:
                        left_top_label2 = np.array([3])
                    elif '1/0' in left_top_label2:
                        left_top_label2 = np.array([2])
                    elif '1/1' in left_top_label2:
                        left_top_label2 = np.array([1])
                    else:
                        left_top_label2 = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('左上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0]
                    if '0/0' in right_top_label:
                        right_top_label = np.array([4])
                    elif '0/1' in right_top_label:
                        right_top_label = np.array([3])
                    elif '1/0' in right_top_label:
                        right_top_label = np.array([2])
                    elif '1/1' in right_top_label:
                        right_top_label = np.array([1])
                    else:
                        right_top_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0]
                    if '0/0' in left_center_label:
                        left_center_label = np.array([4])
                    elif '0/1' in left_center_label:
                        left_center_label = np.array([3])
                    elif '1/0' in left_center_label:
                        left_center_label = np.array([2])
                    elif '1/1' in left_center_label:
                        left_center_label = np.array([1])
                    else:
                        left_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0]
                    if '0/0' in right_center_label:
                        right_center_label = np.array([4])
                    elif '0/1' in right_center_label:
                        right_center_label = np.array([3])
                    elif '1/0' in right_center_label:
                        right_center_label = np.array([2])
                    elif '1/1' in right_center_label:
                        right_center_label = np.array([1])
                    else:
                        right_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0]
                    if '0/0' in left_bottom_label:
                        left_bottom_label = np.array([4])
                    elif '0/1' in left_bottom_label:
                        left_bottom_label = np.array([3])
                    elif '1/0' in left_bottom_label:
                        left_bottom_label = np.array([2])
                    elif '1/1' in left_bottom_label:
                        left_bottom_label = np.array([1])
                    else:
                        left_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0]
                    if '0/0' in right_bottom_label:
                        right_bottom_label = np.array([4])
                    elif '0/1' in right_bottom_label:
                        right_bottom_label = np.array([3])
                    elif '1/0' in right_bottom_label:
                        right_bottom_label = np.array([2])
                    elif '1/1' in right_bottom_label:
                        right_bottom_label = np.array([1])
                    else:
                        right_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)
                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(1 for x in values1 if x == 4 or x == 3)
                    if count1 <= 4:
                        mixed_label1 = np.array([1., 0.])
                    else:
                        mixed_label1 = np.array([0., 1.])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(1 for x in values2 if x == 4 or x == 3)
                    if count2 <= 4:
                        mixed_label2 = np.array([1., 0.])
                    else:
                        mixed_label2 = np.array([0., 1.])
                # else:
                #     if 'Health' in line2:
                #         choosen_index = 1
                #         img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                #         image2 = Image.open(img_path2).convert('RGB')
                #         img_size = image2.size[0]
                #         mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                #         mask_image2_2 = Image.open(mask_img_path2)
                #         mask_new_subimage_org2 = np.asarray(mask_image2_2)
                #         mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                #         mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                #         mask_new_left_subimage_org2[:, mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                #                                                                                 :,
                #                                                                                 mask_new_subimage_org2.shape[
                #                                                                                     1] // 2:]
                #         mask_new_right_subimage_org2[:, :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                #                                                                                  :, :
                #                                                                                     mask_new_subimage_org2.shape[
                #                                                                                         1] // 2]
                #
                #         org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                #         org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                #         image2_np = np.asarray(image2)
                #         image_np = np.asarray(image)
                #
                #         mixed_img1 = np.expand_dims(
                #             mask_new_left_subimage_org2 // 255,
                #             axis=-1) * image2_np + np.expand_dims(
                #             org_right_mask // 255,
                #             axis=-1) * image_np
                #         mixed_img2 = np.expand_dims(
                #             mask_new_right_subimage_org2 // 255,
                #             axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                #                                                   axis=-1) * image_np
                #         mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                #         mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                #         mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                #         mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')
                #
                #         mixed_img1 = Image.fromarray(mixed_img1)
                #         mixed_img2 = Image.fromarray(mixed_img2)
                #
                #
                #         values1 = [
                #             subregions_labels[0],  # 0
                #             subregions_labels[2],  # 1
                #             subregions_labels[4],  # 0
                #         ]
                #         count1 = sum(1 for x in values1 if x == 4 or x == 3)
                #         if count1 <= 1:
                #             mixed_label2 = np.array([1., 0.])
                #         else:
                #             mixed_label2 = np.array([0., 1.])
                #
                #         values2 = [
                #             subregions_labels[1],  # 0
                #             subregions_labels[3],  # 1
                #             subregions_labels[5],  # 0
                #         ]
                #         count2 = sum(1 for x in values2 if x == 4 or x == 3)
                #         if count2 <= 1:
                #             mixed_label1 = np.array([1., 0.])
                #         else:
                #             mixed_label1 = np.array([0., 1.])

        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)

        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])

            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])

            mixed_img1, mixed_mask1s = self.transforms(mixed_img1, mixed_mask1s)
            mixed_img2, mixed_mask2s = self.transforms(mixed_img2, mixed_mask2s)
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask1s[0])
            # plt.show()
            # plt.imshow(mixed_mask1s[-1])
            # plt.show()
            # plt.imshow(mixed_mask1s[-2])
            # plt.show()
            # plt.imshow(mixed_img2)
            # plt.show()
            # plt.imshow(mixed_mask2s[0])
            # plt.show()
            # plt.imshow(mixed_mask2s[-1])
            # plt.show()
            # plt.imshow(mixed_mask2s[-2])
            # plt.show()
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1s)
            masks.append(mixed_mask1s)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                # labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_masks1=[]
            mixed_masks2 = []
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            mixed_masks1.append(mixed_mask1)
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))

            mixed_masks2.append(mixed_mask2)
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            masks.append(mixed_masks1)

            masks.append(mixed_masks2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                subregions_labels.append(np.array([-1]))
                mixed_sublabels.append(np.array([-1]))
                mixed_sublabels2.append(np.array([-1]))
                # labels.append(np.array([-1,-1]))
        return imgs, masks, labels, subregions_labels, mixed_sublabels, mixed_sublabels2, masks_index, imgname

import matplotlib.patches as patches
class Shanxi_w7masks_Subregions_5classes_Mixednew_temp_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_Mixednew_temp_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        subregions_labels_mixed = []
        subregions_labels_mixed2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_save_path='/disk3/wjr/dataset/nejm/example_image/'+imgname.split('.png')[0]+'/'
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(img_save_path+ 'img.png', bbox_inches='tight')
        plt.show()

        plt.imshow(mask_image2, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(img_save_path+ 'mask.png', bbox_inches='tight')
        plt.show()

        aa=np.array(mask_image2)/255.
        bb=np.array(image)
        mixed_img1=aa[...,None]*bb
        mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
        plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(img_save_path + 'masked_img.png', bbox_inches='tight')
        plt.show()

        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        # # 假设 image 是你的原图（numpy数组或PIL图像）
        # plt.imshow(image, cmap='gray')
        # # 创建矩形对象
        # rect = patches.Rectangle(
        #     (xmin, ymin),  # 左上角坐标
        #     width, height,  # 宽和高
        #     linewidth=2,  # 线条粗细
        #     edgecolor='r',  # 边框颜色（红色）
        #     facecolor='none'  # 填充颜色（透明）
        # )
        # # 将矩形添加到图像上
        # plt.gca().add_patch(rect)  # gca() 获取当前坐标轴
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()
        if csv_line.size != 0:

            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]

            rect1 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )
            mixed_img1 = Image.fromarray(mask_new_subimage_lefttop.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_lefttop.png', bbox_inches='tight')
            plt.show()
            plt.imshow(left_top_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'lefttopimg.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'lefttopmask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(left_top_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'lefttopmaskedimg.png', bbox_inches='tight')
            plt.show()
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)


            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]
            mixed_img1 = Image.fromarray(mask_new_subimage_righttop.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_righttop.png', bbox_inches='tight')
            plt.show()
            plt.imshow(right_top_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'righttopimg.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'righttopmask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(right_top_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'righttopmaskedimg.png', bbox_inches='tight')
            plt.show()
            rect2 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )


            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
            mixed_img1 = Image.fromarray(mask_new_subimage_leftcenter.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_leftcenter.png', bbox_inches='tight')
            plt.show()
            plt.imshow(left_center_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftcenterimg.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftcentermask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(left_center_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftcentermaskedimg.png', bbox_inches='tight')
            plt.show()
            rect3 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )


            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]
            mixed_img1 = Image.fromarray(mask_new_subimage_rightcenter.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_rightcenter.png', bbox_inches='tight')
            plt.show()
            plt.imshow(right_center_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightcenterimg.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightcentermask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(right_center_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightcentermaskedimg.png', bbox_inches='tight')
            plt.show()
            rect4 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_leftbottom.png', bbox_inches='tight')
            plt.show()
            plt.imshow(left_bottom_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftbottom.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftbottommask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(left_bottom_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'leftbottommaskedimg.png', bbox_inches='tight')
            plt.show()
            rect5 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]
            mixed_img1 = Image.fromarray(mask_new_subimage_rightbottom.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'masked_rightbottom.png', bbox_inches='tight')
            plt.show()
            plt.imshow(right_bottom_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightbottomimg.png', bbox_inches='tight')
            plt.show()
            plt.imshow(F.crop(mask_image2, ymin, xmin, height, width), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightbottommask.png', bbox_inches='tight')
            plt.show()
            aa = np.array(F.crop(mask_image2, ymin, xmin, height, width)) / 255.
            bb = np.array(right_bottom_img)
            mixed_img1 = aa[..., None] * bb
            mixed_img1 = Image.fromarray(mixed_img1.astype(np.uint8))
            plt.imshow(mixed_img1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'rightbottommaskedimg.png', bbox_inches='tight')
            plt.show()
            rect6 = patches.Rectangle(
                (xmin, ymin),  # 左上角坐标
                width, height,  # 宽和高
                linewidth=2,  # 线条粗细
                edgecolor='r',  # 边框颜色（红色）
                facecolor='none'  # 填充颜色（透明）
            )
            # # # 将矩形添加到图像上
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            plt.gca().add_patch(rect1)  # gca() 获取当前坐标轴
            plt.gca().add_patch(rect2)  # gca() 获取当前坐标轴
            plt.gca().add_patch(rect3)  # gca() 获取当前坐标轴
            plt.gca().add_patch(rect4)  # gca() 获取当前坐标轴
            plt.gca().add_patch(rect5)  # gca() 获取当前坐标轴
            plt.gca().add_patch(rect6)  # gca() 获取当前坐标轴
            plt.axis('off')  # 隐藏坐标轴
            plt.savefig(img_save_path + 'img_wboxes.png', bbox_inches='tight')
            plt.show()




            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('右上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0]
                    if '0/0' in left_top_label2:
                        left_top_label2 = np.array([4])
                    elif '0/1' in left_top_label2:
                        left_top_label2 = np.array([3])
                    elif '1/0' in left_top_label2:
                        left_top_label2 = np.array([2])
                    elif '1/1' in left_top_label2:
                        left_top_label2 = np.array([1])
                    else:
                        left_top_label2 = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('左上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0]
                    if '0/0' in right_top_label:
                        right_top_label = np.array([4])
                    elif '0/1' in right_top_label:
                        right_top_label = np.array([3])
                    elif '1/0' in right_top_label:
                        right_top_label = np.array([2])
                    elif '1/1' in right_top_label:
                        right_top_label = np.array([1])
                    else:
                        right_top_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0]
                    if '0/0' in left_center_label:
                        left_center_label = np.array([4])
                    elif '0/1' in left_center_label:
                        left_center_label = np.array([3])
                    elif '1/0' in left_center_label:
                        left_center_label = np.array([2])
                    elif '1/1' in left_center_label:
                        left_center_label = np.array([1])
                    else:
                        left_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0]
                    if '0/0' in right_center_label:
                        right_center_label = np.array([4])
                    elif '0/1' in right_center_label:
                        right_center_label = np.array([3])
                    elif '1/0' in right_center_label:
                        right_center_label = np.array([2])
                    elif '1/1' in right_center_label:
                        right_center_label = np.array([1])
                    else:
                        right_center_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0]
                    if '0/0' in left_bottom_label:
                        left_bottom_label = np.array([4])
                    elif '0/1' in left_bottom_label:
                        left_bottom_label = np.array([3])
                    elif '1/0' in left_bottom_label:
                        left_bottom_label = np.array([2])
                    elif '1/1' in left_bottom_label:
                        left_bottom_label = np.array([1])
                    else:
                        left_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0]
                    if '0/0' in right_bottom_label:
                        right_bottom_label = np.array([4])
                    elif '0/1' in right_bottom_label:
                        right_bottom_label = np.array([3])
                    elif '1/0' in right_bottom_label:
                        right_bottom_label = np.array([2])
                    elif '1/1' in right_bottom_label:
                        right_bottom_label = np.array([1])
                    else:
                        right_bottom_label = np.array([0])
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)
                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(1 for x in values1 if x == 4 or x == 3)
                    if count1 <= 4:
                        mixed_label1 = np.array([1., 0.])
                    else:
                        mixed_label1 = np.array([0., 1.])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(1 for x in values2 if x == 4 or x == 3)
                    if count2 <= 4:
                        mixed_label2 = np.array([1., 0.])
                    else:
                        mixed_label2 = np.array([0., 1.])
                # else:
                #     if 'Health' in line2:
                #         choosen_index = 1
                #         img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                #         image2 = Image.open(img_path2).convert('RGB')
                #         img_size = image2.size[0]
                #         mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                #         mask_image2_2 = Image.open(mask_img_path2)
                #         mask_new_subimage_org2 = np.asarray(mask_image2_2)
                #         mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                #         mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                #         mask_new_left_subimage_org2[:, mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                #                                                                                 :,
                #                                                                                 mask_new_subimage_org2.shape[
                #                                                                                     1] // 2:]
                #         mask_new_right_subimage_org2[:, :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                #                                                                                  :, :
                #                                                                                     mask_new_subimage_org2.shape[
                #                                                                                         1] // 2]
                #
                #         org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                #         org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                #         image2_np = np.asarray(image2)
                #         image_np = np.asarray(image)
                #
                #         mixed_img1 = np.expand_dims(
                #             mask_new_left_subimage_org2 // 255,
                #             axis=-1) * image2_np + np.expand_dims(
                #             org_right_mask // 255,
                #             axis=-1) * image_np
                #         mixed_img2 = np.expand_dims(
                #             mask_new_right_subimage_org2 // 255,
                #             axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                #                                                   axis=-1) * image_np
                #         mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                #         mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                #         mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                #         mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')
                #
                #         mixed_img1 = Image.fromarray(mixed_img1)
                #         mixed_img2 = Image.fromarray(mixed_img2)
                #
                #
                #         values1 = [
                #             subregions_labels[0],  # 0
                #             subregions_labels[2],  # 1
                #             subregions_labels[4],  # 0
                #         ]
                #         count1 = sum(1 for x in values1 if x == 4 or x == 3)
                #         if count1 <= 1:
                #             mixed_label2 = np.array([1., 0.])
                #         else:
                #             mixed_label2 = np.array([0., 1.])
                #
                #         values2 = [
                #             subregions_labels[1],  # 0
                #             subregions_labels[3],  # 1
                #             subregions_labels[5],  # 0
                #         ]
                #         count2 = sum(1 for x in values2 if x == 4 or x == 3)
                #         if count2 <= 1:
                #             mixed_label1 = np.array([1., 0.])
                #         else:
                #             mixed_label1 = np.array([0., 1.])

        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)

        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])

            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])

            mixed_img1, mixed_mask1s = self.transforms(mixed_img1, mixed_mask1s)
            mixed_img2, mixed_mask2s = self.transforms(mixed_img2, mixed_mask2s)
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask1s[0])
            # plt.show()
            # plt.imshow(mixed_mask1s[-1])
            # plt.show()
            # plt.imshow(mixed_mask1s[-2])
            # plt.show()
            # plt.imshow(mixed_img2)
            # plt.show()
            # plt.imshow(mixed_mask2s[0])
            # plt.show()
            # plt.imshow(mixed_mask2s[-1])
            # plt.show()
            # plt.imshow(mixed_mask2s[-2])
            # plt.show()
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1s)
            masks.append(mixed_mask1s)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                # labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_masks1=[]
            mixed_masks2 = []
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            mixed_masks1.append(mixed_mask1)
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))
            mixed_masks1.append(torch.zeros_like(mixed_mask1))

            mixed_masks2.append(mixed_mask2)
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            mixed_masks2.append(torch.zeros_like(mixed_mask2))
            masks.append(mixed_masks1)

            masks.append(mixed_masks2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                subregions_labels.append(np.array([-1]))
                mixed_sublabels.append(np.array([-1]))
                mixed_sublabels2.append(np.array([-1]))
                # labels.append(np.array([-1,-1]))
        return imgs, masks, labels, subregions_labels, mixed_sublabels, mixed_sublabels2, masks_index, imgname
class Shanxi_w7masks_Subregions_5classes_MixedHealthnew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_5classes_MixedHealthnew_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        subregions_labels_mixed = []
        subregions_labels_mixed2 = []
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                if 'Health' in imgname2:
                    csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                    if csv_line2.size != 0:
                        choosen_index = 1
                        img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                        image2 = Image.open(img_path2).convert('RGB')
                        img_size = image2.size[0]
                        mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                        mask_image2_2 = Image.open(mask_img_path2)
                        mask_new_subimage_org2 = np.asarray(mask_image2_2)
                        mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                        mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                        mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                        left_upper_index2 = csv_line2.columns.get_loc('右上')
                        next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                        values = csv_line2[next_three_columns2].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_top_label2 = values[0]
                        if '0/0' in left_top_label2:
                            left_top_label2 = np.array([4])
                        elif '0/1' in left_top_label2:
                            left_top_label2 = np.array([3])
                        elif '1/0' in left_top_label2:
                            left_top_label2 = np.array([2])
                        elif '1/1' in left_top_label2:
                            left_top_label2 = np.array([1])
                        else:
                            left_top_label2 = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                        mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                               int(ymin):int(ymax),
                                                                                               int(xmin):int(xmax)]
                        subregions_imgs2.append(left_top_img2)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(left_top_label2)

                        left_upper_index = csv_line2.columns.get_loc('左上')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_top_label = values[0]
                        if '0/0' in right_top_label:
                            right_top_label = np.array([4])
                        elif '0/1' in right_top_label:
                            right_top_label = np.array([3])
                        elif '1/0' in right_top_label:
                            right_top_label = np.array([2])
                        elif '1/1' in right_top_label:
                            right_top_label = np.array([1])
                        else:
                            right_top_label = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        right_top_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(right_top_img)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(right_top_label)
                        mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                                int(ymin):int(ymax),
                                                                                                int(xmin):int(xmax)]

                        # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                        # plt.show()
                        left_upper_index = csv_line2.columns.get_loc('右中')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_center_label = values[0]
                        if '0/0' in left_center_label:
                            left_center_label = np.array([4])
                        elif '0/1' in left_center_label:
                            left_center_label = np.array([3])
                        elif '1/0' in left_center_label:
                            left_center_label = np.array([2])
                        elif '1/1' in left_center_label:
                            left_center_label = np.array([1])
                        else:
                            left_center_label = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_center_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(left_center_img)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_labels2.append(left_center_label)
                        mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('左中')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_center_label = values[0]
                        if '0/0' in right_center_label:
                            right_center_label = np.array([4])
                        elif '0/1' in right_center_label:
                            right_center_label = np.array([3])
                        elif '1/0' in right_center_label:
                            right_center_label = np.array([2])
                        elif '1/1' in right_center_label:
                            right_center_label = np.array([1])
                        else:
                            right_center_label = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        right_center_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_imgs2.append(right_center_img)
                        subregions_labels2.append(right_center_label)
                        mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('右下')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        left_bottom_label = values[0]
                        if '0/0' in left_bottom_label:
                            left_bottom_label = np.array([4])
                        elif '0/1' in left_bottom_label:
                            left_bottom_label = np.array([3])
                        elif '1/0' in left_bottom_label:
                            left_bottom_label = np.array([2])
                        elif '1/1' in left_bottom_label:
                            left_bottom_label = np.array([1])
                        else:
                            left_bottom_label = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_imgs2.append(left_bottom_img)
                        subregions_labels2.append(left_bottom_label)
                        mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                        left_upper_index = csv_line2.columns.get_loc('左下')
                        next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                        values = csv_line2[next_three_columns].values
                        xmin = values[0, 1] / 1024 * img_size
                        ymin = values[0, 2] / 1024 * img_size
                        xmax = values[0, 3] / 1024 * img_size
                        ymax = values[0, 4] / 1024 * img_size
                        right_bottom_label = values[0]
                        if '0/0' in right_bottom_label:
                            right_bottom_label = np.array([4])
                        elif '0/1' in right_bottom_label:
                            right_bottom_label = np.array([3])
                        elif '1/0' in right_bottom_label:
                            right_bottom_label = np.array([2])
                        elif '1/1' in right_bottom_label:
                            right_bottom_label = np.array([1])
                        else:
                            right_bottom_label = np.array([0])
                        height = ymax - ymin
                        width = xmax - xmin
                        right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                        subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                        subregions_imgs2.append(right_bottom_img)
                        subregions_labels2.append(right_bottom_label)
                        mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]
                        image2_np = np.asarray(image2)
                        image_np = np.asarray(image)
                        mixed_img1 = np.expand_dims((
                                                            mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                    axis=-1) * image2_np + np.expand_dims(
                            (
                                    mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                            axis=-1) * image_np
                        mixed_img2 = np.expand_dims(
                            (
                                    mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                            axis=-1) * image_np + np.expand_dims(
                            (
                                    mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                            axis=-1) * image2_np

                        mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                        mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                        mixed_img1 = Image.fromarray(mixed_img1)
                        mixed_img2 = Image.fromarray(mixed_img2)
                        # plt.imshow(image)
                        # plt.show()
                        # plt.imshow(image2)
                        # plt.show()
                        # plt.imshow(mixed_img1)
                        # plt.show()
                        # plt.imshow(mixed_img2)
                        # plt.show()
                        values1 = [
                            subregions_labels2[1],  # 0
                            subregions_labels2[3],  # 1
                            subregions_labels2[5],  # 40
                            subregions_labels[0],  # 0
                            subregions_labels[2],  # 1
                            subregions_labels[4],  # 0
                        ]
                        count1 = sum(1 for x in values1 if x == 4 or x == 3)
                        if count1 <= 4:
                            mixed_label1 = np.array([1., 0.])
                        else:
                            mixed_label1 = np.array([0., 1.])

                        values2 = [
                            subregions_labels2[0],  # 0
                            subregions_labels2[2],  # 1
                            subregions_labels2[4],  # 40
                            subregions_labels[1],  # 0
                            subregions_labels[3],  # 1
                            subregions_labels[5],  # 0
                        ]
                        count2 = sum(1 for x in values2 if x == 4 or x == 3)
                        if count2 <= 4:
                            mixed_label2 = np.array([1., 0.])
                        else:
                            mixed_label2 = np.array([0., 1.])
                    else:
                        if 'Health' in line2:
                            choosen_index = 1
                            img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                            image2 = Image.open(img_path2).convert('RGB')
                            img_size = image2.size[0]
                            mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                            mask_image2_2 = Image.open(mask_img_path2)
                            mask_new_subimage_org2 = np.asarray(mask_image2_2)
                            mask_new_left_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                            mask_new_right_subimage_org2 = np.zeros_like(mask_new_subimage_org2)
                            mask_new_left_subimage_org2[:,
                            mask_new_subimage_org2.shape[1] // 2:] = mask_new_subimage_org2[
                                                                     :,
                                                                     mask_new_subimage_org2.shape[
                                                                         1] // 2:]
                            mask_new_right_subimage_org2[:,
                            :mask_new_subimage_org2.shape[1] // 2] = mask_new_subimage_org2[
                                                                     :, :
                                                                        mask_new_subimage_org2.shape[
                                                                            1] // 2]

                            org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                            org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop
                            image2_np = np.asarray(image2)
                            image_np = np.asarray(image)

                            mixed_img1 = np.expand_dims(
                                mask_new_left_subimage_org2 // 255,
                                axis=-1) * image2_np + np.expand_dims(
                                org_right_mask // 255,
                                axis=-1) * image_np
                            mixed_img2 = np.expand_dims(
                                mask_new_right_subimage_org2 // 255,
                                axis=-1) * image2_np + np.expand_dims(org_left_mask // 255,
                                                                      axis=-1) * image_np
                            mixed_mask1 = mask_new_left_subimage_org2 + org_right_mask
                            mixed_mask2 = mask_new_right_subimage_org2 + org_left_mask
                            mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                            mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                            mixed_img1 = Image.fromarray(mixed_img1)
                            mixed_img2 = Image.fromarray(mixed_img2)

                            values1 = [
                                subregions_labels[0],  # 0
                                subregions_labels[2],  # 1
                                subregions_labels[4],  # 0
                            ]
                            count1 = sum(1 for x in values1 if x == 4 or x == 3)
                            if count1 <= 1:
                                mixed_label2 = np.array([1., 0.])
                            else:
                                mixed_label2 = np.array([0., 1.])

                            values2 = [
                                subregions_labels[1],  # 0
                                subregions_labels[3],  # 1
                                subregions_labels[5],  # 0
                            ]
                            count2 = sum(1 for x in values2 if x == 4 or x == 3)
                            if count2 <= 1:
                                mixed_label1 = np.array([1., 0.])
                            else:
                                mixed_label1 = np.array([0., 1.])


        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images = self.transforms(image, mask_all_images)

        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])



            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])

            mixed_img1, mixed_mask1s = self.transforms(mixed_img1, mixed_mask1s)
            mixed_img2, mixed_mask2s = self.transforms(mixed_img2, mixed_mask2s)
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask1s[0])
            # plt.show()
            # plt.imshow(mixed_mask1s[-1])
            # plt.show()
            # plt.imshow(mixed_mask1s[-2])
            # plt.show()
            # plt.imshow(mixed_img2)
            # plt.show()
            # plt.imshow(mixed_mask2s[0])
            # plt.show()
            # plt.imshow(mixed_mask2s[-1])
            # plt.show()
            # plt.imshow(mixed_mask2s[-2])
            # plt.show()
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1s)
            masks.append(mixed_mask1s)
            labels.append(mixed_label1)
            labels.append(mixed_label2)
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                # labels.append(subregions_labels[ii])

            # aa=1
        else:
            mixed_img1 = torch.zeros_like(image)
            mixed_mask1 = torch.zeros_like(mask_all_images[0])
            mixed_img2 = torch.zeros_like(image)
            mixed_mask2 = torch.zeros_like(mask_all_images[0])
            imgs.append(mixed_img1)
            imgs.append(mixed_img2)
            masks.append(mixed_mask1)
            masks.append(mixed_mask2)
            labels.append(np.array([-1,-1]))
            labels.append(np.array([-1,-1]))
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                subregions_labels.append(np.array([-1]))
                mixed_sublabels.append(np.array([-1]))
                mixed_sublabels2.append(np.array([-1]))
                # labels.append(np.array([-1,-1]))
        return imgs, masks, labels, subregions_labels, mixed_sublabels, mixed_sublabels2, masks_index, imgname
class Shanxi_wmasks_Subregionsroi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmasks_Subregionsroi_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        imgs=[]
        masks=[]
        labels=[]
        sub_rois=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024*img_size
            ymin = values[0, 2] / 1024*img_size
            xmax = values[0, 3] / 1024*img_size
            ymax = values[0, 4] / 1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label = np.array([0., 1.])
            else:
                left_top_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)

            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label = np.array([0., 1.])
            else:
                right_top_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label = np.array([0., 1.])
            else:
                left_center_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label = np.array([0., 1.])
            else:
                right_center_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label = np.array([0., 1.])
            else:
                left_bottom_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label = np.array([0., 1.])
            else:
                right_bottom_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]



        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_labels!=[]:
            for ii in range(len(subregions_labels)):
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                labels.append(np.array([-1,-1]))
                sub_rois.append(torch.zeros(4))
        return imgs, masks, labels, sub_rois, masks_index, imgname
class Shanxi_wmasks_Subregionsroi5classes_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmasks_Subregionsroi5classes_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        imgs=[]
        masks=[]
        labels=[]
        sub_rois=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024*img_size
            ymin = values[0, 2] / 1024*img_size
            xmax = values[0, 3] / 1024*img_size
            ymax = values[0, 4] / 1024*img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            else:
                left_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)

            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            else:
                right_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            else:
                left_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            else:
                right_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            else:
                left_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            else:
                right_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]



        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        sub_roi_labels = []
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_all_images)
        labels.append(label)
        if subregions_labels!=[]:
            for ii in range(len(subregions_labels)):
                sub_roi_labels.append(subregions_labels[ii])
        else:
            for ii in range(6):
                sub_roi_labels.append(np.array([-1]))
                sub_rois.append(torch.zeros(4))
        return imgs, masks, labels, sub_rois, sub_roi_labels, masks_index, imgname
class Shanxi_2wmasks_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 maskimgpath2,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_2wmasks_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.maskimgpath2 = maskimgpath2
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        masks_index=[]
        imgs=[]
        masks=[]
        masks2 = []
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_img_path2 = os.path.join(self.maskimgpath2, imgname.split('.png')[0] + '.png')
        if os.path.exists(mask_img_path2) and os.path.isfile(mask_img_path2):
            mask_image = Image.open(mask_img_path2)
            mask_image2 = Image.open(mask_img_path)
            # plt.imshow(mask_image)
            # plt.show()
            # plt.imshow(mask_image2)
            # plt.show()
            masks_index.append(2)
        else:
            mask_image = Image.open(mask_img_path)
            mask_image2 = Image.open(mask_img_path)
            masks_index.append(1)


        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)

        image, mask_image, mask_image2 = self.transforms(image, mask_image, mask_image2)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_image)
        masks2.append(mask_image2)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                masks2.append(maska)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                masks2.append(maska)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, masks, masks2, labels, masks_index, imgname
class Shanxi_w2masks_5Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 maskimgpath2,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w2masks_5Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.maskimgpath2 = maskimgpath2
        self.csvpath=csvpath

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
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_fine_labels = []
        subregions_masks=[]
        masks_index=[]
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_img_path2 = os.path.join(self.maskimgpath2, imgname.split('.png')[0] + '.png')
        if os.path.exists(mask_img_path2) and os.path.isfile(mask_img_path2):
            mask_image = Image.open(mask_img_path2)
            mask_image2 = Image.open(mask_img_path)
            masks_index.append(2)
        else:
            mask_image = Image.open(mask_img_path)
            mask_image2 = Image.open(mask_img_path)
            masks_index.append(1)


        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0, 0]
            if '0/0' in left_top_label or '0/1' in left_top_label:
                left_top_label=np.array([0.,1.])
            else:
                left_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0, 0]
            if '0/0' in right_top_label or '0/1' in right_top_label:
                right_top_label=np.array([0.,1.])
            else:
                right_top_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label or '0/1' in left_center_label:
                left_center_label=np.array([0.,1.])
            else:
                left_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)

            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0, 0]
            if '0/0' in right_center_label or '0/1' in right_center_label:
                right_center_label=np.array([0.,1.])
            else:
                right_center_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0, 0]
            if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
                left_bottom_label=np.array([0.,1.])
            else:
                left_bottom_label =  np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
                right_bottom_label=np.array([0.,1.])
            else:
                right_bottom_label = np.array([1., 0.])
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        imgs.append(image)
        masks.append(mask_image)
        labels.append(label)
        if subregions_imgs!=[]:
            for ii in range(len(subregions_imgs)):
                # image, mask = self.local_transfo_mask(image, mask)
                imga, maska = self.data_subregions_transform(subregions_imgs[ii], subregions_masks[ii])
                imgs.append(imga)
                masks.append(maska)
                labels.append(subregions_labels[ii])
            # aa=1
        else:
            for ii in range(6):
                imga = torch.zeros([3,self.sub_img_size,self.sub_img_size])
                maska = torch.zeros([1, self.sub_img_size, self.sub_img_size])
                imgs.append(imga)
                masks.append(maska)
                labels.append(np.array([-1,-1]))
            # aa=2

        return imgs, masks, labels, masks_index, imgname


import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """固定所有随机种子，确保实验可重复"""
    random.seed(seed)  # 固定Python随机数生成器
    np.random.seed(seed)  # 固定NumPy随机数生成器
    torch.manual_seed(seed)  # 固定PyTorch CPU随机数生成器
    torch.cuda.manual_seed(seed)  # 固定PyTorch GPU随机数生成器
    torch.cuda.manual_seed_all(seed)  # 多GPU环境下固定所有GPU的随机数生成器

    # 以下设置可提高实验结果的确定性，但可能影响性能
    torch.backends.cudnn.deterministic = True  # 固定CuDNN的确定性模式
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动调优功能
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定Python哈希种子
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import transforms as pth_transform

    set_seed(42)  # 可以选择任意整数作为种子值
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
        pth_transform.RandomResizedCrop(256, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        pth_transform.RandomHorizontalFlip(p=0.5),
        pth_transform.RandomRotation(degrees=(-15, 15)),
        pth_transform.RandomAutocontrast(p=0.3),
        pth_transform.RandomEqualize(p=0.3),
        # pth_transform.ToTensor(),
        # pth_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset=Shanxi_w7masks_Subregions_5classes_Mixednew_temp_Dataset('/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_img_1024',
                                              '/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_mask_1024',
                                              txtpath='/disk3/wjr/dataset/nejm/shanxidataset/dataaug_vis.txt',
                                   csvpath='/disk3/wjr/dataset/nejm/shanxidataset/subregions_label_shanxi_new.xlsx',
                                   data_transform=global_transfo2, data_subregions_transform=global_transfo2_subregions,
                                   sub_img_size=256,)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    for sample in data_loader:
        print(1)
