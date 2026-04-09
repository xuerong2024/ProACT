
import torch
torch.set_num_threads(1)
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

from matplotlib.font_manager import FontProperties
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    # 设置散点颜色
    colors = [ 'r', 'blue','#e38c7a','#656667', 'cyan',  '#99a4bc', 'lime',  'violet', 'm', 'peru', 'olivedrab',
              'hotpink']
    # 设置散点形状
    maker = ['^', '*', 's','o', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 绘制散点图
    plt.scatter(data[label == 0, 0], data[label == 0, 1], c='r', label='Sick', cmap='brg', marker='^', edgecolors='r', alpha=0.65)
    plt.scatter(data[label == 1, 0], data[label == 1, 1], c='b', label='Normal', cmap='brg', marker='*', edgecolors='b', alpha=0.65)
    # 显示图例
    # font = FontProperties(family='serif', size=14)
    font = FontProperties(family='Times New Roman', size=14)
    plt.legend(prop=font, handletextpad=0.01)
    # for i in range(data.shape[0]):
    #     plt.scatter(data[i, 0], data[i, 1], cmap='brg', marker=maker[label[i]], c=colors[label[i]], edgecolors=colors[label[i]], alpha=0.65)
    #
    #
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=colors[label[i]+2],
    #              fontdict={'weight': 'bold', 'size': 5})
    plt.xticks([])
    plt.yticks([])
    # plt.legend(loc='best', prop={'size': 12, 'color': 'r', 'marker':'^'}, title='Sick')
    # plt.title(title)
    # 关闭外围边界线框
    # plt.axis('off')
    # 调整外围边界，删除多余空白
    plt.axis('tight')
    plt.tight_layout()
    return fig

import os
import time
import json
import random
import argparse
import datetime
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 添加自定义路径到全局模块搜索路径中
sys.path.append('/disk3/wjr/workspace/sec_nejm/nejm_baseline_wjr/')
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.configv2 import get_config
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from mmpretrain import get_model
from utils.datasets_pneum import Guiyang_Feiqu_Dataset, Shanxi_Dataset, Shanxi_Subregions_Dataset, Fibrosis_Dataset
print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)
import utils.misc as misc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda:4')
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
def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_option():
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
    parser.add_argument('--output', default='/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/normalandsubregions_cls_augmentation/512/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
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

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--data_root', type=str, default='/disk3/wjr/dataset/nejm/shanxidataset/', help='The path of dataset')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    args, unparsed = parser.parse_known_args()
    return args

from torchvision import transforms as pth_transforms

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


def evaluate_unsupervised_clustering(X, y_true):
    """
    对特征矩阵进行无监督聚类（K-means），并计算与真实标签的分类准确性。
    """
    # 1. 使用K-means进行聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # 2. 计算混淆矩阵
    # 初始化混淆矩阵（真实标签行 × 预测标签列）
    confusion_matrix = np.zeros((2, 2))
    for t, p in zip(y_true, y_pred):
        confusion_matrix[int(t), int(p)] += 1

    # 3. 使用匈牙利算法找到最佳标签映射
    # maximize=True 表示最大化匹配数
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)
    label_mapping = {col: row for col, row in zip(col_ind, row_ind)}

    # 4. 重新映射预测标签
    mapped_y_pred = np.array([label_mapping[p] for p in y_pred])

    # 5. 计算准确率
    accuracy = (mapped_y_pred == y_true).mean()

    return accuracy
def main(config):
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    transform = pth_transforms.Compose([
        pth_transforms.Resize((512, 512)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = Shanxi_Dataset(args.data_root + 'seg_rec_img_1024',
                                 txtpath=args.data_root + 'sick_health_5fold_txt/fold' + str(args.experi) + '_val.txt',
                                 data_transform=transform)
    dataset_test = Shanxi_Dataset(args.data_root + 'seg_rec_img_1024',
                                  txtpath=args.data_root + 'sick_health_5fold_txt/fold' + str(
                                      args.experi) + '_test.txt', data_transform=transform)
    dataset_test2 = Shanxi_Dataset('/disk3/wjr/dataset/nejm/guizhoudataset/seg_rec_img_1024',
                                   txtpath='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one.txt',
                                   data_transform=transform)

    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank,
                                                       shuffle=False)
    sampler_test2 = torch.utils.data.DistributedSampler(dataset_test2, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2, sampler=sampler_test2,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    pretrained_cfg = '/disk3/wjr/workspace_lyt/pretrained_weight/mmpre/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
    model = get_model('swin-tiny_16xb64_in1k', pretrained=pretrained_cfg)
    model.head.fc = nn.Linear(768, 2, bias=True)

    model.to(device)
    tsne_path = os.path.join(config.OUTPUT, 'tsne')
    if not os.path.exists(tsne_path):
        os.makedirs(tsne_path)

    ckp_path = os.path.join(config.OUTPUT, f'model_best.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        feas = []
        tsne_labels = []
        for idx, (images, targets, imgname) in enumerate(data_loader_val):
            imgname=imgname[0]
            images = images.to(device)
            labels = targets.to(device)
            fea = model.backbone(images)[0]
            fea = model.neck(fea).squeeze(0).cpu().detach().numpy()
            pred = model(images)
            if 'Sick' in imgname:
                tsne_labels.append(0)
            else:
                tsne_labels.append(1)
            fea_mean = fea.mean()
            fea_std = fea.std()
            fea = (fea - fea_mean) / fea_std
            feas.append(fea)
        tsne_labels = np.asarray(tsne_labels)
        feas = np.asarray(feas)
        n_samples, n_features = feas.shape
        tsne = TSNE(n_components=2, init='random')
        result = tsne.fit_transform(feas)
        # print('result.shape', result.shape)
        plot_embedding(result, tsne_labels, 'Shanxi Val t-SNE embedding of the features.')
        tsne_path = os.path.join(config.OUTPUT, f'tsne/shanxi_val.png')
        plt.savefig(tsne_path)
        # plt.show()
        plt.close()

        feas = []
        tsne_labels = []
        for idx, (images, targets, imgname) in enumerate(data_loader_test):
            imgname = imgname[0]
            images = images.to(device)
            labels = targets.to(device)
            fea = model.backbone(images)[0]
            fea = model.neck(fea).squeeze(0).cpu().detach().numpy()
            pred = model(images)
            if 'Sick' in imgname:
                tsne_labels.append(0)
            else:
                tsne_labels.append(1)
            fea_mean = fea.mean()
            fea_std = fea.std()
            fea = (fea - fea_mean) / fea_std
            feas.append(fea)
        tsne_labels = np.asarray(tsne_labels)
        feas = np.asarray(feas)
        n_samples, n_features = feas.shape
        tsne = TSNE(n_components=2, init='random')
        result = tsne.fit_transform(feas)
        # print('result.shape', result.shape)
        plot_embedding(result, tsne_labels, 'Shanxi test t-SNE embedding of the features.')
        tsne_path = os.path.join(config.OUTPUT, f'tsne/shanxi_test.png')
        plt.savefig(tsne_path)
        # plt.show()
        plt.close()

        feas = []
        tsne_labels = []
        for idx, (images, targets, imgname) in enumerate(data_loader_test2):
            imgname = imgname[0]
            images = images.to(device)
            labels = targets.to(device)
            fea = model.backbone(images)[0]
            fea = model.neck(fea).squeeze(0).cpu().detach().numpy()
            pred = model(images)
            if 'Sick' in imgname:
                tsne_labels.append(0)
            else:
                tsne_labels.append(1)
            fea_mean = fea.mean()
            fea_std = fea.std()
            fea = (fea - fea_mean) / fea_std
            feas.append(fea)
        tsne_labels = np.asarray(tsne_labels)
        feas = np.asarray(feas)
        n_samples, n_features = feas.shape
        tsne = TSNE(n_components=2, init='random')
        result = tsne.fit_transform(feas)
        # print('result.shape', result.shape)
        plot_embedding(result, tsne_labels, 'Guizhou t-SNE embedding of the features.')
        tsne_path = os.path.join(config.OUTPUT, f'tsne/guizhou_test.png')
        plt.savefig(tsne_path)
        # plt.show()
        plt.close()



if __name__ == '__main__':
    import gc
    import pandas as pd

    torch.set_num_threads(3)
    lrs = [8e-5]
    batch_sizes = [16]
    model_name = ['swintiny']
    for kk in range(len(lrs)):
        for jj in range(len(model_name)):
            shanxi_test_avgs = []
            shanxi_test_accs = []
            shanxi_test_aucs = []
            shanxi_test_sens = []
            shanxi_test_spec = []

            guizhou_test_avgs = []
            guizhou_test_accs = []
            guizhou_test_aucs = []
            guizhou_test_sens = []
            guizhou_test_spec = []

            val_aucs = []

            # for ii in range(len(seeds)):
            for ii in range(5):
                ii += 1
                avgs = []
                lr = lrs[kk]
                args = parse_option()
                args.batch_size = batch_sizes[0]
                args.base_lr = lr
                args.experi = str(ii)
                args.model_name = model_name[jj]
                config = get_config(args)
                config.OUTPUT = os.path.join(args.output, args.model_name,
                                             str(args.experi) + 'experi_' + args.model_name + '_lr' + str(args.base_lr))

                config.freeze()

                os.makedirs(config.OUTPUT, exist_ok=True)
                logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

                path = os.path.join(config.OUTPUT, "config.json")
                with open(path, "w") as f:
                    f.write(config.dump())
                logger.info(f"Full config saved to {path}")

                # print config
                logger.info(config.dump())
                logger.info(json.dumps(vars(args)))
                main(config)

