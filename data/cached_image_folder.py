# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image

import os
import zipfile
import io
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files"""
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    len(os.path.splitext(file_foler_name)[-1]) == 0 and \
                    file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        if extension is None:
            extension = ['.*']
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1:])

        return file_lists

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        try:
            im = Image.open(io.BytesIO(data))
        except:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix,im_file_name), class_index)

            images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no"):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            # samples = make_dataset_with_ann(os.path.join(root, ann_file),
            #                                 os.path.join(root, img_prefix),
            #                                 extensions)
            samples = make_dataset_with_ann(ann_file,
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample // 10) == 0:
                t = time.time() - start_time
                print(f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full":
                samples_bytes[index] = (ZipReader.read(path), target)
            elif self.cache_mode == "part" and index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class DatasetMaskFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, mask_root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no"):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.mask_root = mask_root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample // 10) == 0:
                t = time.time() - start_time
                print(f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full":
                samples_bytes[index] = (ZipReader.read(path), target)
            elif self.cache_mode == "part" and index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CachedImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
import numpy as np
class CachedImageMaskFolder(DatasetMaskFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mask_root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedImageMaskFolder, self).__init__(root, mask_root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples
        self.root=root
        self.mask_root=mask_root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        targets=[]
        path, target = self.samples[index]
        image = self.loader(path)
        mask_path=path.replace(self.root, self.mask_root)
        mask = Image.open(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        if self.target_transform is not None:
            target = self.target_transform(target)
        for ii in range(len(image)):
            targets.append(target)
        return image, mask, targets
import random
class CachedMixedImageMaskFolder(DatasetMaskFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mask_root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedMixedImageMaskFolder, self).__init__(root, mask_root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples
        self.root=root
        self.mask_root=mask_root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        targets=[]
        path, target = self.samples[index]
        image = self.loader(path)
        mask_path=path.replace(self.root, self.mask_root)
        mask = Image.open(mask_path)
        choosen_index = 0
        while (choosen_index) == 0:
            randomidx = random.randint(0, len(self.samples) - 1)
            path2, target2 = self.samples[randomidx]
            if 'NORMAL' in path2:
                choosen_index = 1
                mixed_image = self.loader(path2)
                mask_path2 = path2.replace(self.root, self.mask_root)
                mixed_mask_image = Image.open(mask_path2)

        if self.transform is not None:
            # image, mask = self.transform(image, mask)
            image = self.transform(image, mask, mixed_image, mixed_mask_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        for ii in range(len(image)):
            targets.append(target)
        return image, targets
class CachedMixedNihImageMaskFolder(DatasetMaskFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mask_root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedMixedNihImageMaskFolder, self).__init__(root, mask_root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples
        self.root=root
        self.mask_root=mask_root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        targets=[]
        path, target = self.samples[index]
        image = self.loader(path)
        mask_path=path.replace(self.root, self.mask_root)
        mask = Image.open(mask_path)
        choosen_index = 0
        while (choosen_index) == 0:
            randomidx = random.randint(0, len(self.samples) - 1)
            path2, target2 = self.samples[randomidx]
            #7代表健康
            if target2==7:
                choosen_index = 1
                mixed_image = self.loader(path2)
                mask_path2 = path2.replace(self.root, self.mask_root)
                mixed_mask_image = Image.open(mask_path2)

        if self.transform is not None:
            # image, mask = self.transform(image, mask)
            image = self.transform(image, mask, mixed_image, mixed_mask_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        for ii in range(len(image)):
            targets.append(target)
        return image, targets
if __name__ == '__main__':
    import torch
    from torchvision.transforms import transforms
    from torchvision.transforms import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize([224,224], interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = CachedMixedImageMaskFolder(root='/disk3/wjr/dataset/CPNXray/segimg',
                                          mask_root='/disk3/wjr/dataset/CPNXray/seg_rec_mask', ann_file='val.txt',
                                          transform=transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=12,
        pin_memory=True,
        shuffle=True
    )
    for idx, (inp, target) in enumerate(val_loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        aa=1