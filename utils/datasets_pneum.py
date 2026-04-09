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
class Shanxi_wmask_feiqu2_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 subimgpath,
                 submaskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_feiqu2_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.subimgpath = subimgpath
        self.submaskimgpath = submaskimgpath
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
        self.samples = []
        for line in self.lines:
            imgname0 = line.strip()  # 去掉换行符
            if ':' in imgname0:
                is_sub=True
                imgname=imgname0.split(':')[0]
                label=int(imgname0.split(':')[-1].split('/')[0])
                if label>0:
                    label=1
                img_path = os.path.join(subimgpath, imgname.split('.png')[0] + '.png')
                mask_path = os.path.join(subimgpath, imgname.split('.png')[0] + '.png')
            else:
                is_sub = False
                imgname=imgname0
                img_path = os.path.join(imgpath, imgname0.split('.png')[0] + '.png')
                mask_path = os.path.join(maskimgpath, imgname0.split('.png')[0] + '.png')
                # 判断 label
                label = 1 if 'Sick' in imgname0 else 0
            # ✅ 存储 (img_path, label, imgname)
            self.samples.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'label': label,
                'imgname': imgname,
                'is_sub': is_sub
            })

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        imgname=sample_info['imgname']
        img_path = sample_info['img_path']
        mask_path = sample_info['mask_path']
        label = sample_info['label']
        is_sub = sample_info['is_sub']
        image = Image.open(img_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        image, mask_image = self.transforms(image, mask_image)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname, is_sub
class Shanxi_wmask_feiqu_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 subimgpath,
                 submaskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_feiqu_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.subimgpath = subimgpath
        self.submaskimgpath = submaskimgpath
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
        self.samples = []
        for line in self.lines:
            imgname0 = line.strip()  # 去掉换行符
            imgname = imgname0.split(':')[0]
            label = imgname0.split(':')[-1].split(',')[0]
            if '0/0' in label:
                label=0
            elif '0/1' in label:
                label=1
            elif '1/0' in label:
                label=2
            elif '1/1' in label:
                label=3
            else:
                label=4

            img_path = os.path.join(subimgpath, imgname.split('.png')[0] + '.png')
            mask_path = os.path.join(subimgpath, imgname.split('.png')[0] + '.png')
            # ✅ 存储 (img_path, label, imgname)
            self.samples.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'label': label,
                'imgname': imgname
            })

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        imgname=sample_info['imgname']
        img_path = sample_info['img_path']
        mask_path = sample_info['mask_path']
        label = sample_info['label']
        image = Image.open(img_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        image, mask_image = self.transforms(image, mask_image)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname
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
        random.seed(1)
        random.shuffle(self.lines)
        # print(self.lines[:5])

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
        # label.append('Health' in labelname)
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname
class Shanxi_wmask_Dataset_2Aug(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 data_transform_aug=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset_2Aug, self).__init__()

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
        self.transforms_aug = data_transform_aug
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        random.seed(1)
        random.shuffle(self.lines)
        # print(self.lines[:5])

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
        image1, mask_image1 = self.transforms(image, mask_image)
        image_aug, mask_image_aug = self.transforms_aug(image, mask_image)

        label = []
        # label.append('Health' in labelname)
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image1, mask_image1, image_aug, mask_image_aug, label, imgname
class Shanxi_wmask_Dataset2(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 out_transform1=None,
                 out_transform2=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset2, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        self.transforms=data_transform
        self.out_transform1 = out_transform1
        self.out_transform2 = out_transform2
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
        image1, mask_image1 = self.out_transform1(image, mask_image)
        image2, mask_image2 = self.out_transform2(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image1, mask_image1, image2, mask_image2,label, imgname
class Shanxi_wmask_Dataset3(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 out_transform1=None,
                 out_transform2=None,
                 out_transform3=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset3, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        self.transforms=data_transform
        self.out_transform1 = out_transform1
        self.out_transform2 = out_transform2
        self.out_transform3 = out_transform3
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
        image1, mask_image1 = self.out_transform1(image, mask_image)
        image2, mask_image2 = self.out_transform2(image, mask_image)
        image3, mask_image3 = self.out_transform3(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image1, mask_image1, image2, mask_image2, image3, mask_image3, label, imgname

class Shanxi_wmask_Dataset5(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 out_transform1=None,
                 out_transform2=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset3, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        self.transforms=data_transform
        self.out_transform1 = out_transform1
        self.out_transform2 = out_transform2
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
        mask_path = os.path.join(self.maskpath, imgname.split('.png')[0] + '.png')
        mask = Image.open(mask_path)
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path).convert('RGB')
        image, mask, mask_image = self.transforms(image,mask, mask_image)
        image2, mask = self.out_transform2(image,mask)
        image2=image2*mask
        mask_image1 = self.out_transform1(mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image2, mask_image1,label, imgname
class Shanxi_wmask_Dataset6(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 out_transform1=None,
                 out_transform2=None,
                 out_transform3=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset4, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        self.transforms=data_transform
        self.out_transform1 = out_transform1
        self.out_transform2 = out_transform2
        self.out_transform3 = out_transform3
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
        mask_path = os.path.join(self.maskpath, imgname.split('.png')[0] + '.png')
        mask = Image.open(mask_path)
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path).convert('RGB')
        image, mask, mask_image = self.transforms(image,mask, mask_image)
        image2, mask2 = self.out_transform2(image,mask)
        image2=image2*mask2
        image3, mask3 = self.out_transform3(image, mask)
        image3 = image3 * mask3
        mask_image1 = self.out_transform1(mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image2, image3, mask_image1,label, imgname

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
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

        self.csv = pd.read_excel(csvpath)

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
            subregions_labels.append(left_top_label)
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
        # imgs.append(image)
        # masks.append(mask_all_images)
        labels.append(label)
        for jj in range(len(subregions_labels)):
            labels.append(subregions_labels[jj])
        return image, mask_all_images, labels, imgname
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
                 label_noise_radio=0.,
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
class Shanxi_w7masks_Subregionsroi5classes_Mixednew_Dataset(torch.utils.data.Dataset):
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
        super(Shanxi_w7masks_Subregionsroi5classes_Mixednew_Dataset, self).__init__()

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
        subregions_labels=[]
        subregions_labels2=[]
        sub_rois2 = []
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


        ####随机组合图像####
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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
                mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                       int(ymin):int(ymax),
                                                                                       int(xmin):int(xmax)]
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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
                subregions_labels2.append(right_top_label)
                mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                        int(ymin):int(ymax),
                                                                                        int(xmin):int(xmax)]


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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
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
                sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
                sub_rois2.append(sub_roi)
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

        mask_all_images = []
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)

        mask_all_images2_1 = []
        mask_all_images2_1.append(mixed_mask1)
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images2_1.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
        subrois2_1=[]
        subrois2_1.append(sub_rois[0])
        subrois2_1.append(sub_rois2[1])
        subrois2_1.append(sub_rois[2])
        subrois2_1.append(sub_rois2[3])
        subrois2_1.append(sub_rois[4])
        subrois2_1.append(sub_rois2[5])
        mixed_img2_1, mask_all_images2_1, subrois2_1= self.transforms(mixed_img1, mask_all_images2_1, sub_rois=subrois2_1)

        mask_all_images2_2 = []
        mask_all_images2_2.append(mixed_mask2)
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
        mask_all_images2_2.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        subrois2_2 = []
        subrois2_2.append(sub_rois2[0])
        subrois2_2.append(sub_rois[1])
        subrois2_2.append(sub_rois2[2])
        subrois2_2.append(sub_rois[3])
        subrois2_2.append(sub_rois2[4])
        subrois2_2.append(sub_rois[5])
        mixed_img2_2, mask_all_images2_2, subrois2_2 = self.transforms(mixed_img2, mask_all_images2_2,
                                                                       sub_rois=subrois2_2)


        sub_roi_labels = []
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        imgs.append(image)
        imgs.append(mixed_img2_1)
        imgs.append(mixed_img2_2)
        masks.append(mask_all_images)
        masks.append(mask_all_images2_1)
        masks.append(mask_all_images2_2)
        labels.append(label)
        labels.append(mixed_label1)
        labels.append(mixed_label2)
        if subregions_labels != []:
            for ii in range(len(subregions_labels)):
                sub_roi_labels.append(subregions_labels[ii])
        else:
            for ii in range(6):
                sub_roi_labels.append(np.array([-1]))
                sub_rois.append(torch.zeros(4))
        return imgs, masks, labels, sub_rois, sub_roi_labels, subrois2_1, values1, subrois2_2, values2, masks_index, imgname
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
class Shanxi_w7masks_Subregions_wsubroi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_Subregions_wsubroi_Dataset, self).__init__()

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
        self.pil2tensor_transform=pil2tensor_transform
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
        sub_rois = []
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
            elif 'unknown' in left_top_label:
                left_top_label=np.array([-1,-1])
            else:
                left_top_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_top_label:
                right_top_label = np.array([-1, -1])
            else:
                right_top_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in left_center_label:
                left_center_label = np.array([-1, -1])
            else:
                left_center_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_center_label:
                right_center_label = np.array([-1, -1])
            else:
                right_center_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in left_bottom_label:
                left_bottom_label = np.array([-1, -1])
            else:
                left_bottom_label =  np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_bottom_label:
                right_bottom_label = np.array([-1, -1])
            else:
                right_bottom_label = np.array([1., 0.])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        mask_image2=mask_all_images[0]
        # image, mask_all_images = self.transforms(image, mask_all_images)
        # print(imgname)

        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label


        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage= F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)
        return imgs, masks, labels, sub_rois_new, masks_index, imgname
class Shanxi_Submasks_Subregions_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 subimgpath,
                 submaskimgpath,
                 txtpath,
                 data_subregions_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_Submasks_Subregions_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.subimgpath = subimgpath
        self.submaskimgpath = submaskimgpath
        self.txtpath = txtpath
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
        subregions_masks=[]
        imgname=line.split('\n')[0].split('.png')[0]
        labelname = imgname.split('_')[0]
        left_top_img_path=os.path.join(self.subimgpath, imgname + '_left_top.png')
        right_top_img_path=os.path.join(self.subimgpath, imgname + '_right_top.png')
        left_top_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_top.png')
        right_top_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_top.png')
        left_top_img = Image.open(left_top_img_path).convert('RGB')
        left_top_mask = Image.open(left_top_mask_path)
        imga, maska = self.data_subregions_transform(left_top_img, left_top_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))

        right_top_img = Image.open(right_top_img_path).convert('RGB')
        right_top_mask = Image.open(right_top_mask_path)
        imga, maska = self.data_subregions_transform(right_top_img, right_top_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))


        left_center_img_path = os.path.join(self.subimgpath, imgname + '_left_center.png')
        right_center_img_path = os.path.join(self.subimgpath, imgname + '_right_center.png')
        left_center_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_center.png')
        right_center_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_center.png')
        left_center_img = Image.open(left_center_img_path).convert('RGB')
        left_center_mask = Image.open(left_center_mask_path)
        imga, maska = self.data_subregions_transform(left_center_img, left_center_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        right_center_img = Image.open(right_center_img_path).convert('RGB')
        right_center_mask = Image.open(right_center_mask_path)
        imga, maska = self.data_subregions_transform(right_center_img, right_center_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))

        left_bottom_img_path = os.path.join(self.subimgpath, imgname + '_left_bottom.png')
        right_bottom_img_path = os.path.join(self.subimgpath, imgname + '_right_bottom.png')
        left_bottom_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_bottom.png')
        right_bottom_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_bottom.png')
        left_bottom_img = Image.open(left_bottom_img_path).convert('RGB')
        left_bottom_mask = Image.open(left_bottom_mask_path)
        imga, maska = self.data_subregions_transform(left_bottom_img, left_bottom_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        right_bottom_img = Image.open(right_bottom_img_path).convert('RGB')
        right_bottom_mask = Image.open(right_bottom_mask_path)
        imga, maska = self.data_subregions_transform(right_bottom_img, right_bottom_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))



        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        # label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        subregions_imgs=torch.cat(subregions_imgs, dim=0)
        subregions_masks = torch.cat(subregions_masks, dim=0)
        return subregions_imgs, subregions_masks, label, imgname


class Shanxi_Submasks_Subregions_Dataset5(torch.utils.data.Dataset):
    def __init__(self,
                 subimgpath,
                 submaskimgpath,
                 txtpath,
                 subtxtpath,
                 data_subregions_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_Submasks_Subregions_Dataset5, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.subimgpath = subimgpath
        self.submaskimgpath = submaskimgpath
        self.txtpath = txtpath
        self.subtxtpath = subtxtpath
        self.data_subregions_transform = data_subregions_transform
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
        line = self.lines[idx]
        subregions_imgs = []
        subregions_masks = []
        subregions_targets = []
        imgnames=[]
        imgname = line.split('\n')[0].split('.png')[0]
        labelname = imgname.split('_')[0]
        find_index=0
        with open(self.subtxtpath, 'r', encoding='gbk') as file_sub:
            for line in file_sub:
                if line.startswith(imgname + '_left_top'):
                    find_index+=1
                    left_top_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in left_top_label:
                        left_top_label = 0
                    elif '0/1' in left_top_label:
                        left_top_label = 1
                    elif '1/0' in left_top_label:
                        left_top_label = 2
                    elif '1/1' in left_top_label:
                        left_top_label = 3
                    else:
                        left_top_label = 4
                elif line.startswith(imgname + '_right_top'):
                    find_index += 1
                    right_top_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in right_top_label:
                        right_top_label = 0
                    elif '0/1' in right_top_label:
                        right_top_label = 1
                    elif '1/0' in right_top_label:
                        right_top_label = 2
                    elif '1/1' in right_top_label:
                        right_top_label = 3
                    else:
                        right_top_label = 4
                elif line.startswith(imgname + '_left_center'):
                    find_index += 1
                    left_center_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in left_center_label:
                        left_center_label = 0
                    elif '0/1' in left_center_label:
                        left_center_label = 1
                    elif '1/0' in left_center_label:
                        left_center_label = 2
                    elif '1/1' in left_center_label:
                        left_center_label = 3
                    else:
                        left_center_label = 4
                elif line.startswith(imgname + '_right_center'):
                    find_index += 1
                    right_center_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in right_center_label:
                        right_center_label = 0
                    elif '0/1' in right_center_label:
                        right_center_label = 1
                    elif '1/0' in right_center_label:
                        right_center_label = 2
                    elif '1/1' in right_center_label:
                        right_center_label = 3
                    else:
                        right_center_label = 4
                elif line.startswith(imgname + '_left_bottom'):
                    find_index += 1
                    left_bottom_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in left_bottom_label:
                        left_bottom_label = 0
                    elif '0/1' in left_bottom_label:
                        left_bottom_label = 1
                    elif '1/0' in left_bottom_label:
                        left_bottom_label = 2
                    elif '1/1' in left_bottom_label:
                        left_bottom_label = 3
                    else:
                        left_bottom_label = 4
                elif line.startswith(imgname + '_right_bottom'):
                    find_index += 1
                    right_bottom_label = line.split(':', 1)[1].split(',')[0]
                    if '0/0' in right_bottom_label:
                        right_bottom_label = 0
                    elif '0/1' in right_bottom_label:
                        right_bottom_label = 1
                    elif '1/0' in right_bottom_label:
                        right_bottom_label = 2
                    elif '1/1' in right_bottom_label:
                        right_bottom_label = 3
                    else:
                        right_bottom_label = 4
                if find_index ==6:
                    break

                # break  # 如果只想找第一个匹配的行，可以在这里跳出循环


        left_top_img_path = os.path.join(self.subimgpath, imgname + '_left_top.png')
        right_top_img_path = os.path.join(self.subimgpath, imgname + '_right_top.png')
        left_top_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_top.png')
        right_top_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_top.png')
        left_top_img = Image.open(left_top_img_path).convert('RGB')
        left_top_mask = Image.open(left_top_mask_path)
        imga, maska = self.data_subregions_transform(left_top_img, left_top_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        subregions_targets.append(torch.tensor(left_top_label).unsqueeze(0))
        imgnames.append(imgname + '_left_top')
        right_top_img = Image.open(right_top_img_path).convert('RGB')
        right_top_mask = Image.open(right_top_mask_path)
        imga, maska = self.data_subregions_transform(right_top_img, right_top_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        subregions_targets.append(torch.tensor(right_top_label).unsqueeze(0))
        imgnames.append(imgname + '_right_top')
        left_center_img_path = os.path.join(self.subimgpath, imgname + '_left_center.png')
        right_center_img_path = os.path.join(self.subimgpath, imgname + '_right_center.png')
        left_center_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_center.png')
        right_center_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_center.png')
        left_center_img = Image.open(left_center_img_path).convert('RGB')
        left_center_mask = Image.open(left_center_mask_path)
        imga, maska = self.data_subregions_transform(left_center_img, left_center_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        imgnames.append(imgname + '_left_center')
        subregions_targets.append(torch.tensor(left_center_label).unsqueeze(0))
        right_center_img = Image.open(right_center_img_path).convert('RGB')
        right_center_mask = Image.open(right_center_mask_path)
        imga, maska = self.data_subregions_transform(right_center_img, right_center_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        subregions_targets.append(torch.tensor(right_center_label).unsqueeze(0))
        imgnames.append(imgname + '_right_center')
        left_bottom_img_path = os.path.join(self.subimgpath, imgname + '_left_bottom.png')
        right_bottom_img_path = os.path.join(self.subimgpath, imgname + '_right_bottom.png')
        left_bottom_mask_path = os.path.join(self.submaskimgpath, imgname + '_left_bottom.png')
        right_bottom_mask_path = os.path.join(self.submaskimgpath, imgname + '_right_bottom.png')
        left_bottom_img = Image.open(left_bottom_img_path).convert('RGB')
        left_bottom_mask = Image.open(left_bottom_mask_path)
        imga, maska = self.data_subregions_transform(left_bottom_img, left_bottom_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        subregions_targets.append(torch.tensor(left_bottom_label).unsqueeze(0))
        imgnames.append(imgname + '_left_bottom')
        right_bottom_img = Image.open(right_bottom_img_path).convert('RGB')
        right_bottom_mask = Image.open(right_bottom_mask_path)
        imga, maska = self.data_subregions_transform(right_bottom_img, right_bottom_mask)
        subregions_imgs.append(imga.unsqueeze(0))
        subregions_masks.append(maska.unsqueeze(0))
        subregions_targets.append(torch.tensor(right_bottom_label).unsqueeze(0))
        imgnames.append(imgname + '_right_bottom')
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        # label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        subregions_imgs = torch.cat(subregions_imgs, dim=0)
        subregions_masks = torch.cat(subregions_masks, dim=0)
        subregions_targets = torch.cat(subregions_targets, dim=0)
        return subregions_imgs, subregions_masks, subregions_targets, label, imgname

class Shanxi_w7masks_5Subregions_wsubroi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Dataset, self).__init__()

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
        self.pil2tensor_transform=pil2tensor_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()
        random.seed(1)
        random.shuffle(self.lines)
        # print(self.lines[:5])

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
        sub_rois = []
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
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            elif 'unknown' in left_top_label:
                left_top_label=np.array([-1])
            else:
                left_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_top_label:
                right_top_label = np.array([-1])
            else:
                right_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            elif 'unknown' in left_center_label:
                left_center_label = np.array([-1])
            else:
                left_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_center_label:
                right_center_label = np.array([-1])
            else:
                right_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in left_bottom_label:
                left_bottom_label = np.array([-1])
            else:
                left_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            elif 'unknown' in right_bottom_label:
                right_bottom_label = np.array([-1])
            else:
                right_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        mask_image2=mask_all_images[0]
        # image, mask_all_images = self.transforms(image, mask_all_images)
        # print(imgname)

        label = []
        # label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label


        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage= F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)
        return imgs, masks, labels, sub_rois_new, masks_index, imgname

class Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset, self).__init__()

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
        self.pil2tensor_transform = pil2tensor_transform

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
        sub_rois = []
        sub_rois2 = []
        imgs=[]
        masks=[]
        labels=[]
        mixed_imgs1 = []
        mixed_masks1 = []
        mixed_labels1 = []
        mixed_imgs2 = []
        mixed_masks2 = []
        mixed_labels2 = []
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask_image2)
        # plt.show()

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
            elif 'unknown' in left_top_label:
                left_top_label=np.array([-1])
            else:
                left_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            # plt.imshow(left_top_img)
            # plt.show()
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
            elif 'unknown' in right_top_label:
                right_top_label=np.array([-1])
            else:
                right_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_top_img)
            # plt.show()
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
            elif 'unknown' in left_center_label:
                left_center_label = np.array([-1])
            else:
                left_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_center_label:
                right_center_label = np.array([-1])
            else:
                right_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in left_bottom_label:
                left_bottom_label = np.array([-1])
            else:
                left_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            elif 'unknown' in right_bottom_label:
                right_bottom_label = np.array([-1])
            else:
                right_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mask_image2_2)
                    # plt.show()
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
                    elif 'unknown' in left_top_label2:
                        left_top_label2 = np.array([-1])
                    else:
                        left_top_label2 = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_top_img2)
                    # plt.show()
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
                    elif 'unknown' in right_top_label:
                        right_top_label = np.array([-1])
                    else:
                        right_top_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]


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
                    elif 'unknown' in left_center_label:
                        left_center_label = np.array([-1])
                    else:
                        left_center_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    elif 'unknown' in right_center_label:
                        right_center_label = np.array([-1])
                    else:
                        right_center_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    elif 'unknown' in left_bottom_label:
                        left_bottom_label = np.array([-1])
                    else:
                        left_bottom_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    elif 'unknown' in right_bottom_label:
                        right_bottom_label = np.array([-1])
                    else:
                        right_bottom_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    mixed_subrois1=[sub_rois[0].clone(),sub_rois2[1].clone(),sub_rois[2].clone(),sub_rois2[3].clone(),sub_rois[4].clone(),sub_rois2[5].clone()]


                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np
                    mixed_subrois2 = [sub_rois2[0].clone(), sub_rois[1].clone(), sub_rois2[2].clone(), sub_rois[3].clone(), sub_rois2[4].clone(), sub_rois[5].clone()]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)

                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_mask1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    # plt.imshow(mixed_mask2)
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
                        mixed_label1 = np.array([0])
                    else:
                        mixed_label1 = np.array([1])

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
                        mixed_label2 = np.array([0])
                    else:
                        mixed_label2 = np.array([1])

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
        mask_image2 = mask_all_images[0]
        # plt.imshow(image)
        # plt.show()
        label = []
        # label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)

        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage = F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            # plt.imshow(mask_new_subimage)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)


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
            # plt.imshow(mixed_img1)
            # plt.show()

            mixed_img1, mixed_mask1s, mixed_subrois1 = self.transforms(mixed_img1, mixed_mask1s, sub_rois=mixed_subrois1)
            mixed_mask_image = mixed_mask1s[0]
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask_image)
            # plt.show()
            for ii in range(len(mixed_subrois1)):
                [xmin, ymin, xmax, ymax] = mixed_subrois1[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img1, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                # plt.imshow(mask_new_subimage)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs1.append(imga)
                mixed_masks1.append(maska)
                mixed_labels1.append(mixed_sublabels[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois1_new = self.pil2tensor_transform(mixed_img1, mixed_mask1s,
                                                                                           sub_rois=mixed_subrois1)

            mixed_imgs1.insert(0, mixed_image_tensor)
            mixed_masks1.insert(0, mixed_mask_all_images_tensor)
            mixed_labels1.insert(0, mixed_label1)


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
            mixed_img2, mixed_mask2s, mixed_subrois2 = self.transforms(mixed_img2, mixed_mask2s,
                                                                       sub_rois=mixed_subrois2)
            # plt.imshow(mixed_img1)
            # plt.show()
            mixed_mask_image2 = mixed_mask2s[0]
            for ii in range(len(mixed_subrois2)):
                [xmin, ymin, xmax, ymax] = mixed_subrois2[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img2, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image2, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs2.append(imga)
                mixed_masks2.append(maska)
                mixed_labels2.append(mixed_sublabels2[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois2_new = self.pil2tensor_transform(mixed_img2,
                                                                                                             mixed_mask2s,
                                                                                                             sub_rois=mixed_subrois2)

            mixed_imgs2.insert(0, mixed_image_tensor)
            mixed_masks2.insert(0, mixed_mask_all_images_tensor)
            mixed_labels2.insert(0, mixed_label2)
        return [imgs, mixed_imgs1, mixed_imgs2], [masks, mixed_masks1, mixed_masks2], [labels, mixed_labels1, mixed_labels2], [sub_rois_new,mixed_subrois1_new,mixed_subrois2_new],masks_index, imgname
class Shanxi_w7masks_5Subregions_wsubroi_Mixednew_onlysicksub_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Mixednew_onlysicksub_Dataset, self).__init__()

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
        self.pil2tensor_transform = pil2tensor_transform

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
        sub_rois = []
        sub_rois2 = []
        imgs=[]
        masks=[]
        labels=[]
        mixed_imgs1 = []
        mixed_masks1 = []
        mixed_labels1 = []
        mixed_imgs2 = []
        mixed_masks2 = []
        mixed_labels2 = []
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask_image2)
        # plt.show()

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
            if 'Sick' in labelname:
                if '0/0' in left_top_label:
                    left_top_label = np.array([4])
                elif '0/1' in left_top_label:
                    left_top_label = np.array([3])
                elif '1/0' in left_top_label:
                    left_top_label = np.array([2])
                elif '1/1' in left_top_label:
                    left_top_label = np.array([1])
                elif 'unknown' in left_top_label:
                    left_top_label = np.array([-1])
                else:
                    left_top_label = np.array([0])
            else:
                left_top_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            # plt.imshow(left_top_img)
            # plt.show()
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
            if 'Sick' in labelname:
                if '0/0' in right_top_label:
                    right_top_label = np.array([4])
                elif '0/1' in right_top_label:
                    right_top_label = np.array([3])
                elif '1/0' in right_top_label:
                    right_top_label = np.array([2])
                elif '1/1' in right_top_label:
                    right_top_label = np.array([1])
                elif 'unknown' in right_top_label:
                    right_top_label = np.array([-1])
                else:
                    right_top_label = np.array([0])
            else:
                right_top_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_top_img)
            # plt.show()
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
            if 'Sick' in labelname:
                if '0/0' in left_center_label:
                    left_center_label = np.array([4])
                elif '0/1' in left_center_label:
                    left_center_label = np.array([3])
                elif '1/0' in left_center_label:
                    left_center_label = np.array([2])
                elif '1/1' in left_center_label:
                    left_center_label = np.array([1])
                elif 'unknown' in left_center_label:
                    left_center_label = np.array([-1])
                else:
                    left_center_label = np.array([0])
            else:
                left_center_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            if 'Sick' in labelname:
                if '0/0' in right_center_label:
                    right_center_label = np.array([4])
                elif '0/1' in right_center_label:
                    right_center_label = np.array([3])
                elif '1/0' in right_center_label:
                    right_center_label = np.array([2])
                elif '1/1' in right_center_label:
                    right_center_label = np.array([1])
                elif 'unknown' in right_center_label:
                    right_center_label = np.array([-1])
                else:
                    right_center_label = np.array([0])
            else:
                right_center_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            if 'Sick' in labelname:
                if '0/0' in left_bottom_label:
                    left_bottom_label = np.array([4])
                elif '0/1' in left_bottom_label:
                    left_bottom_label = np.array([3])
                elif '1/0' in left_bottom_label:
                    left_bottom_label = np.array([2])
                elif '1/1' in left_bottom_label:
                    left_bottom_label = np.array([1])
                elif 'unknown' in left_bottom_label:
                    left_bottom_label = np.array([-1])
                else:
                    left_bottom_label = np.array([0])
            else:
                left_bottom_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
            if 'Sick' in labelname:
                if '0/0' in right_bottom_label:
                    right_bottom_label = np.array([4])
                elif '0/1' in right_bottom_label:
                    right_bottom_label = np.array([3])
                elif '1/0' in right_bottom_label:
                    right_bottom_label = np.array([2])
                elif '1/1' in right_bottom_label:
                    right_bottom_label = np.array([1])
                elif 'unknown' in right_bottom_label:
                    right_bottom_label = np.array([-1])
                else:
                    right_bottom_label = np.array([0])
            else:
                right_bottom_label = np.array([-1])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
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
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mask_image2_2)
                    # plt.show()
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
                    if 'Sick' in imgname2:
                        if '0/0' in left_top_label2:
                            left_top_label2 = np.array([4])
                        elif '0/1' in left_top_label2:
                            left_top_label2 = np.array([3])
                        elif '1/0' in left_top_label2:
                            left_top_label2 = np.array([2])
                        elif '1/1' in left_top_label2:
                            left_top_label2 = np.array([1])
                        elif 'unknown' in left_top_label2:
                            left_top_label2 = np.array([-1])
                        else:
                            left_top_label2 = np.array([0])
                    else:
                        left_top_label2 = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_top_img2)
                    # plt.show()
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
                    if 'Sick' in imgname2:
                        if '0/0' in right_top_label:
                            right_top_label = np.array([4])
                        elif '0/1' in right_top_label:
                            right_top_label = np.array([3])
                        elif '1/0' in right_top_label:
                            right_top_label = np.array([2])
                        elif '1/1' in right_top_label:
                            right_top_label = np.array([1])
                        elif 'unknown' in right_top_label:
                            right_top_label = np.array([-1])
                        else:
                            right_top_label = np.array([0])
                    else:
                        right_top_label = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]


                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0]
                    if 'Sick' in imgname2:
                        if '0/0' in left_center_label:
                            left_center_label = np.array([4])
                        elif '0/1' in left_center_label:
                            left_center_label = np.array([3])
                        elif '1/0' in left_center_label:
                            left_center_label = np.array([2])
                        elif '1/1' in left_center_label:
                            left_center_label = np.array([1])
                        elif 'unknown' in left_center_label:
                            left_center_label = np.array([-1])
                        else:
                            left_center_label = np.array([0])
                    else:
                        left_center_label = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    if 'Sick' in imgname2:
                        if '0/0' in right_center_label:
                            right_center_label = np.array([4])
                        elif '0/1' in right_center_label:
                            right_center_label = np.array([3])
                        elif '1/0' in right_center_label:
                            right_center_label = np.array([2])
                        elif '1/1' in right_center_label:
                            right_center_label = np.array([1])
                        elif 'unknown' in right_center_label:
                            right_center_label = np.array([-1])
                        else:
                            right_center_label = np.array([0])
                    else:
                        right_center_label = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    if 'Sick' in imgname2:
                        if '0/0' in left_bottom_label:
                            left_bottom_label = np.array([4])
                        elif '0/1' in left_bottom_label:
                            left_bottom_label = np.array([3])
                        elif '1/0' in left_bottom_label:
                            left_bottom_label = np.array([2])
                        elif '1/1' in left_bottom_label:
                            left_bottom_label = np.array([1])
                        elif 'unknown' in left_bottom_label:
                            left_bottom_label = np.array([-1])
                        else:
                            left_bottom_label = np.array([0])
                    else:
                        left_bottom_label = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    if 'Sick' in imgname2:
                        if '0/0' in right_bottom_label:
                            right_bottom_label = np.array([4])
                        elif '0/1' in right_bottom_label:
                            right_bottom_label = np.array([3])
                        elif '1/0' in right_bottom_label:
                            right_bottom_label = np.array([2])
                        elif '1/1' in right_bottom_label:
                            right_bottom_label = np.array([1])
                        elif 'unknown' in right_bottom_label:
                            right_bottom_label = np.array([-1])
                        else:
                            right_bottom_label = np.array([0])
                    else:
                        right_bottom_label = np.array([-1])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
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
                    mixed_subrois1=[sub_rois[0].clone(),sub_rois2[1].clone(),sub_rois[2].clone(),sub_rois2[3].clone(),sub_rois[4].clone(),sub_rois2[5].clone()]


                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np
                    mixed_subrois2 = [sub_rois2[0].clone(), sub_rois[1].clone(), sub_rois2[2].clone(), sub_rois[3].clone(), sub_rois2[4].clone(), sub_rois[5].clone()]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)

                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_mask1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    # plt.imshow(mixed_mask2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(1 for x in values1 if x == 0 or x == 1 or x == 2)
                    if count1 <= 1:
                        mixed_label1 = np.array([1])
                    else:
                        mixed_label1 = np.array([0])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(1 for x in values2 if x == 0 or x == 1 or x == 2)
                    if count2 <= 1:
                        mixed_label2 = np.array([1])
                    else:
                        mixed_label2 = np.array([0])

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
        mask_image2 = mask_all_images[0]
        # plt.imshow(image)
        # plt.show()
        label = []
        # label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)

        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage = F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            # plt.imshow(mask_new_subimage)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)


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
            # plt.imshow(mixed_img1)
            # plt.show()

            mixed_img1, mixed_mask1s, mixed_subrois1 = self.transforms(mixed_img1, mixed_mask1s, sub_rois=mixed_subrois1)
            mixed_mask_image = mixed_mask1s[0]
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask_image)
            # plt.show()
            for ii in range(len(mixed_subrois1)):
                [xmin, ymin, xmax, ymax] = mixed_subrois1[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img1, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                # plt.imshow(mask_new_subimage)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs1.append(imga)
                mixed_masks1.append(maska)
                mixed_labels1.append(mixed_sublabels[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois1_new = self.pil2tensor_transform(mixed_img1, mixed_mask1s,
                                                                                           sub_rois=mixed_subrois1)

            mixed_imgs1.insert(0, mixed_image_tensor)
            mixed_masks1.insert(0, mixed_mask_all_images_tensor)
            mixed_labels1.insert(0, mixed_label1)


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
            mixed_img2, mixed_mask2s, mixed_subrois2 = self.transforms(mixed_img2, mixed_mask2s,
                                                                       sub_rois=mixed_subrois2)
            # plt.imshow(mixed_img1)
            # plt.show()
            mixed_mask_image2 = mixed_mask2s[0]
            for ii in range(len(mixed_subrois2)):
                [xmin, ymin, xmax, ymax] = mixed_subrois2[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img2, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image2, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs2.append(imga)
                mixed_masks2.append(maska)
                mixed_labels2.append(mixed_sublabels2[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois2_new = self.pil2tensor_transform(mixed_img2,
                                                                                                             mixed_mask2s,
                                                                                                             sub_rois=mixed_subrois2)

            mixed_imgs2.insert(0, mixed_image_tensor)
            mixed_masks2.insert(0, mixed_mask_all_images_tensor)
            mixed_labels2.insert(0, mixed_label2)
        return [imgs, mixed_imgs1, mixed_imgs2], [masks, mixed_masks1, mixed_masks2], [labels, mixed_labels1, mixed_labels2], [sub_rois_new,mixed_subrois1_new,mixed_subrois2_new],masks_index, imgname

import cv2
class Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset_new(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset_new, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        random.seed(seed)  # 同时固定Python的随机种子
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.csvpath = csvpath
        self.sub_img_size = sub_img_size
        self.csv = pd.read_excel(csvpath)
        self.txtpath = txtpath
        self.label_noise_radio = label_noise_radio

        if data_transform is None:
            # 注意：原代码中的transforms.Compose需要传入两个参数，但这里只有一个，可能需要修正
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transforms = data_transform
        self.data_subregions_transform = data_subregions_transform
        self.pil2tensor_transform = pil2tensor_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

        # 预处理：筛选出CSV中有效的图像名称，避免后续无效检查
        self.valid_img_indices = []
        for idx, line in enumerate(self.lines):
            imgname = line.split('\n')[0]
            if not self.csv.loc[self.csv["胸片名称"] == imgname].empty:
                self.valid_img_indices.append(idx)
        assert len(self.valid_img_indices) > 0, "No valid images found in CSV"

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line = self.lines[idx]
        imgname = line.split('\n')[0]

        # 读取主图像和掩码
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')

        # 使用OpenCV读取图像，提高IO效率
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB

        mask_image2 = cv2.imread(mask_img_path, 0)  # 以灰度模式读取掩码

        # 处理CSV数据
        csv_line = self.csv.loc[self.csv["胸片名称"] == imgname]

        if not csv_line.empty:
            img_size = image.shape[0]  # 使用OpenCV读取的图像，shape为(H, W, C)

            # 提取主图像的子区域
            subregions_imgs, subregions_labels, subregions_masks, sub_rois = self.extract_subregions(
                image, mask_image2, csv_line, img_size)

            # 创建所有掩码的列表（主掩码和6个子区域掩码）
            mask_all_images = []
            mask_all_images.append(self.numpy_to_pil(mask_image2, mode='L'))

            # 为每个子区域创建掩码
            for i in range(6):
                # 创建空白掩码
                mask_subregion = np.zeros_like(mask_image2)
                xmin, ymin, xmax, ymax = sub_rois[i].numpy().astype(int)
                mask_subregion[ymin:ymax, xmin:xmax] = mask_image2[ymin:ymax, xmin:xmax]
                mask_all_images.append(self.numpy_to_pil(mask_subregion, mode='L'))

            # 随机选择另一张有效图像
            random_idx = random.randint(0, len(self.lines) - 1)
            # random_idx = random.choice(self.valid_img_indices)
            line2 = self.lines[random_idx]
            imgname2 = line2.split('\n')[0]
            csv_line2 = self.csv.loc[self.csv["胸片名称"] == imgname2]

            # 读取第二张图像和掩码
            img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
            mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')

            image2 = cv2.imread(img_path2)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            mask_image2_2 = cv2.imread(mask_img_path2, 0)

            # 提取第二张图像的子区域
            subregions_imgs2, subregions_labels2, subregions_masks2, sub_rois2 = self.extract_subregions(
                image2, mask_image2_2, csv_line2, img_size)

            # 生成混合图像
            mixed_img1, mixed_mask1, mixed_subrois1 = self.generate_mixed_image(
                image, image2, mask_image2, mask_image2_2, sub_rois, sub_rois2,
                subregions_labels, subregions_labels2, mix_pattern=1)

            mixed_img2, mixed_mask2, mixed_subrois2 = self.generate_mixed_image(
                image, image2, mask_image2, mask_image2_2, sub_rois, sub_rois2,
                subregions_labels, subregions_labels2, mix_pattern=2)

            # 转换为PIL图像
            image_pil = self.numpy_to_pil(image)

            # 主图像和掩码的变换
            if self.transforms is not None:
                image_tensor, mask_all_images_tensor, sub_rois = self.transforms(
                    image_pil, mask_all_images, sub_rois=sub_rois)
                mask_image2_tensor = mask_all_images_tensor[0]

            # 主标签
            label = np.array(['Health' in imgname]).astype(np.float32)

            # 处理子区域
            imgs = []
            masks = []
            labels = []

            # 添加主图像和标签
            imgs.append(image_tensor)
            masks.append(mask_all_images_tensor)  # masks[0] 包含所有掩码
            labels.append(label)

            # 处理每个子区域
            for ii in range(len(sub_rois)):
                [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(image_tensor, int(ymin), int(xmin), int(height), int(width))
                mask_new_subimage = F.crop(mask_image2_tensor, int(ymin), int(xmin), int(height), int(width))

                if self.data_subregions_transform is not None:
                    imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                else:
                    imga, maska = left_top_img, mask_new_subimage

                imgs.append(imga)
                masks.append(maska)  # masks[1:7] 是子区域掩码
                labels.append(subregions_labels[ii])

            # 应用最终的张量转换
            if self.pil2tensor_transform is not None:
                image_tensor, mask_all_images_tensor, sub_rois = self.pil2tensor_transform(
                    image_tensor, mask_all_images_tensor, sub_rois=sub_rois)
                mask_image2_tensor = mask_all_images_tensor[0]

            # 更新主图像和掩码
            imgs[0] = image_tensor
            masks[0] = mask_all_images_tensor  # 确保更新后的所有掩码

            # 处理混合图像
            mixed_imgs1, mixed_masks1, mixed_labels1, mixed_subrois1 = self.process_mixed_image(
                mixed_img1, mixed_mask1, mixed_subrois1, subregions_labels, subregions_labels2, pattern=1)

            mixed_imgs2, mixed_masks2, mixed_labels2, mixed_subrois2 = self.process_mixed_image(
                mixed_img2, mixed_mask2, mixed_subrois2, subregions_labels, subregions_labels2, pattern=2)

        else:
            # 处理无效CSV行的情况
            masks_index = torch.tensor(1)
            # 返回空值或默认值
            return [], [], [], [], masks_index, imgname

        return [imgs, mixed_imgs1, mixed_imgs2], [masks, mixed_masks1, mixed_masks2], [labels, mixed_labels1,
                                                                                       mixed_labels2], [sub_rois,
                                                                                                        mixed_subrois1,
                                                                                                        mixed_subrois2], torch.tensor(
            2), imgname

    def extract_subregions(self, image, mask_image, csv_line, img_size):
        """提取图像的6个子区域（右上、左上、右中、左中、右下、左下）"""
        subregions_imgs = []
        subregions_labels = []
        subregions_masks = []
        sub_rois = []

        # 定义子区域的名称和对应的CSV列名
        regions = ["右上", "左上", "右中", "左中", "右下", "左下"]

        for region in regions:
            col_idx = csv_line.columns.get_loc(region)
            values = csv_line.iloc[0, col_idx:col_idx + 5].values  # 获取该行数据

            # 提取坐标并缩放
            xmin = values[1] / 1024 * img_size
            ymin = values[2] / 1024 * img_size
            xmax = values[3] / 1024 * img_size
            ymax = values[4] / 1024 * img_size

            # 转换标签
            label_str = values[0]
            label = self.convert_label(label_str)

            # 裁剪子区域
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            sub_img = image[ymin:ymax, xmin:xmax]
            sub_mask = mask_image[ymin:ymax, xmin:xmax]

            # 存储结果
            subregions_imgs.append(sub_img)
            subregions_labels.append(label)
            subregions_masks.append(sub_mask)
            sub_rois.append(torch.tensor([xmin, ymin, xmax, ymax]))

        return subregions_imgs, subregions_labels, subregions_masks, sub_rois

    def convert_label(self, label_str):
        """将标签字符串转换为数值"""
        if '0/0' in label_str:
            return np.array([4])
        elif '0/1' in label_str:
            return np.array([3])
        elif '1/0' in label_str:
            return np.array([2])
        elif '1/1' in label_str:
            return np.array([1])
        elif 'unknown' in label_str:
            return np.array([-1])
        else:
            return np.array([0])

    def generate_mixed_image(self, image, image2, mask1, mask2, sub_rois, sub_rois2, labels1, labels2, mix_pattern=1):
        """生成混合图像和掩码"""
        # 创建空白掩码
        h, w = image.shape[:2]
        mask_new_subimage = np.zeros((h, w), dtype=np.uint8)

        # 根据混合模式选择不同的区域组合
        if mix_pattern == 1:
            # 模式1：左侧来自原图，右侧来自第二张图
            regions_to_merge = [0, 2, 4]  # 右上、右中、右下
            regions_to_merge2 = [1, 3, 5]  # 左上、左中、左下
        else:
            # 模式2：左侧来自第二张图，右侧来自原图
            regions_to_merge = [1, 3, 5]  # 左上、左中、左下
            regions_to_merge2 = [0, 2, 4]  # 右上、右中、右下

        # 合并掩码
        for i in regions_to_merge:
            xmin, ymin, xmax, ymax = sub_rois[i].numpy().astype(int)
            mask_new_subimage[ymin:ymax, xmin:xmax] = mask1[ymin:ymax, xmin:xmax]

        for i in regions_to_merge2:
            xmin, ymin, xmax, ymax = sub_rois2[i].numpy().astype(int)
            mask_new_subimage[ymin:ymax, xmin:xmax] = mask2[ymin:ymax, xmin:xmax]

        # 生成混合图像
        if mix_pattern == 1:
            mixed_img = np.copy(image)
            for i in regions_to_merge2:
                xmin, ymin, xmax, ymax = sub_rois2[i].numpy().astype(int)
                mixed_img[ymin:ymax, xmin:xmax] = image2[ymin:ymax, xmin:xmax]
            mixed_subrois = [sub_rois[0].clone(), sub_rois2[1].clone(), sub_rois[2].clone(), sub_rois2[3].clone(), sub_rois[4].clone(), sub_rois2[5].clone()]
        else:
            mixed_img = np.copy(image2)
            for i in regions_to_merge2:
                xmin, ymin, xmax, ymax = sub_rois[i].numpy().astype(int)
                mixed_img[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
            mixed_subrois = [sub_rois2[0].clone(), sub_rois[1].clone(), sub_rois2[2].clone(), sub_rois[3].clone(), sub_rois2[4].clone(), sub_rois[5].clone()]

        # 确定混合标签
        if mix_pattern == 1:
            values = [labels2[1], labels2[3], labels2[5], labels1[0], labels1[2], labels1[4]]
        else:
            values = [labels2[0], labels2[2], labels2[4], labels1[1], labels1[3], labels1[5]]

        count = sum(1 for x in values if x == 4 or x == 3)
        mixed_label = np.array([0]) if count <= 4 else np.array([1])

        return mixed_img, mask_new_subimage, mixed_subrois

    def process_mixed_image(self, mixed_img, mixed_mask, mixed_subrois, labels1, labels2, pattern=1):
        """处理混合图像，应用变换并提取子区域"""
        # 转换为PIL图像
        mixed_img_pil = self.numpy_to_pil(mixed_img)
        mixed_mask_pil = self.numpy_to_pil(mixed_mask, mode='L')

        # 创建所有掩码的列表
        mask_all_images = []
        mask_all_images.append(mixed_mask_pil)

        # 为每个子区域创建掩码
        for i in range(6):
            # 创建空白掩码
            mask_subregion = np.zeros_like(mixed_mask)
            xmin, ymin, xmax, ymax = mixed_subrois[i].numpy().astype(int)
            mask_subregion[ymin:ymax, xmin:xmax] = mixed_mask[ymin:ymax, xmin:xmax]
            mask_all_images.append(self.numpy_to_pil(mask_subregion, mode='L'))

        # 应用变换
        if self.transforms is not None:
            mixed_img_tensor, mixed_mask_tensors, mixed_subrois = self.transforms(
                mixed_img_pil, mask_all_images, sub_rois=mixed_subrois)
            mixed_mask_tensor = mixed_mask_tensors[0]

        # 确定混合标签
        if pattern == 1:
            values = [labels2[1], labels2[3], labels2[5], labels1[0], labels1[2], labels1[4]]
        else:
            values = [labels2[0], labels2[2], labels2[4], labels1[1], labels1[3], labels1[5]]

        count = sum(1 for x in values if x == 4 or x == 3)
        mixed_label = np.array([0]) if count <= 4 else np.array([1])

        # 处理子区域
        mixed_imgs = []
        mixed_masks = []
        mixed_labels = []

        # 添加完整混合图像和标签
        mixed_imgs.append(mixed_img_tensor)
        mixed_masks.append(mixed_mask_tensors)  # mixed_masks[0] 包含所有掩码
        mixed_labels.append(mixed_label)

        # 处理每个子区域
        for ii in range(len(mixed_subrois)):
            [xmin, ymin, xmax, ymax] = mixed_subrois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            sub_img = F.crop(mixed_img_tensor, int(ymin), int(xmin), int(height), int(width))
            sub_mask = F.crop(mixed_mask_tensor, int(ymin), int(xmin), int(height), int(width))

            if self.data_subregions_transform is not None:
                sub_img, sub_mask = self.data_subregions_transform(sub_img, sub_mask)

            mixed_imgs.append(sub_img)
            mixed_masks.append(sub_mask)  # mixed_masks[1:7] 是子区域掩码

            # 根据模式选择标签
            if pattern == 1:
                # 0,2,4来自原图，1,3,5来自第二张图
                label = labels1[ii] if ii in [0, 2, 4] else labels2[ii]
            else:
                # 0,2,4来自第二张图，1,3,5来自原图
                label = labels2[ii] if ii in [0, 2, 4] else labels1[ii]

            mixed_labels.append(label)

        # 应用最终的张量转换
        if self.pil2tensor_transform is not None:
            mixed_img_tensor, mixed_mask_tensors, mixed_subrois = self.pil2tensor_transform(
                mixed_img_tensor, mixed_mask_tensors, sub_rois=mixed_subrois)
            mixed_mask_tensor = mixed_mask_tensors[0]

            # 更新完整图像和掩码
            mixed_imgs[0] = mixed_img_tensor
            mixed_masks[0] = mixed_mask_tensors  # 确保更新后的所有掩码

        # 返回更新后的子区域坐标
        return mixed_imgs, mixed_masks, mixed_labels, mixed_subrois

    def numpy_to_pil(self, img_np, mode='RGB'):
        """将numpy数组转换为PIL图像"""
        from PIL import Image
        return Image.fromarray(img_np, mode=mode)

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
    ])
    transform = pth_transform.Compose([
        pth_transform.Resize((256, 256)),
        pth_transform.ToTensor(),
        pth_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil2tensor_transfo = pth_transform.Compose([pth_transform.ToTensor(),
                                              pth_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
    dataset=Shanxi_wmask_Dataset3('/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_img_1024',
                                              '/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_mask_1024',
                                              txtpath='/disk3/wjr/dataset/nejm/shanxidataset/dataaug_vis.txt',
                                   data_transform=global_transfo2, out_transform1=transform, out_transform2=pil2tensor_transfo
                                   )

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(data_loader):
        print(1)
