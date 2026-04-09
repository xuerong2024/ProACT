
import pandas as pd
import torch
import numpy as np
from sklearn.utils import compute_class_weight
# 读取 CSV 文件
txt_dir='/disk3/wjr/dataset/shanxi_subregions/txt/txt_5fold/fold5_train.txt'
CLASSES = [ '0_0', '0_1', '1_0', '1_1']
# 准备存储文件名和标签的列表
file_names = []
labels = []

# 读取 test_list.txt 并填充文件名和标签
with open(txt_dir, 'r') as test_file:
    for line in test_file:
        # file_name = line.strip()  # 去掉空白字符
        labelname = line.split('/n')[0]
        file_names.append(labelname)
        label = []
        label.append('0_0' in labelname)
        label.append('0_1' in labelname)
        label.append('1_0' in labelname)
        label.append('1_1' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        labels.append(label)
labels=np.stack(labels, axis=0)
# 计算每个类别的类权重
class_weights = {}
for i in range(len(CLASSES)):
    # 提取每个类别的标签
    y_i = labels[:, i]
    # 计算类权重
    weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_i)
    class_weights[CLASSES[i]] = weight[1]  # 只取存在的类别的权重
    print(CLASSES[i], ':', weight[1])

classes_weights=torch.tensor([4.895682555527826, 4.895682555527826, 4.233275787656452,2.8165111270638907, 9.77672564166459,8.955489614243323,
                 38.05431619786615,10.580906148867314,12.02390438247012,23.215384615384615,21.808782657031685,33.880829015544045,
                 17.215445370776656, 272.4583333333333])
aa=1

