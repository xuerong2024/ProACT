
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
# 读取 CSV 文件
csv_dir = '/disk1/wjr/dataset/NIHorg_dataset/Data_Entry_2017.csv'
txt_dir='/disk3/wjr/workspace/CheXNet-master/ChestX-ray14/labels/train_list.txt'
csv_data = pd.read_csv(csv_dir)
CLASSES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# 创建一个字典以快速查找 Finding Labels
finding_labels_dict = dict(zip(csv_data['Image Index'], csv_data['Finding Labels']))
# 准备存储文件名和标签的列表
file_names = []
labels = []

# 读取 test_list.txt 并填充文件名和标签
with open(txt_dir, 'r') as test_file:
    for line in test_file:
        # file_name = line.strip()  # 去掉空白字符
        file_name = line.split(' ')[0]
        file_names.append(file_name)

        # 获取对应的 Finding Labels
        if file_name in finding_labels_dict:
            finding_labels = finding_labels_dict[file_name]
            labels.append(finding_labels)
        else:
            labels.append('No Finding Labels')  # 处理找不到的情况

label_df = pd.DataFrame({'file': file_names, 'Finding Labels': labels})

# 将 Finding Labels 列拆分为多列
for pathology in CLASSES:
    label_df[pathology] = label_df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

# 排序
label_df = label_df.sort_values(by='file')

# 提取文件名和标签
files = label_df['file'].values.tolist()
labels = label_df[CLASSES].values
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

