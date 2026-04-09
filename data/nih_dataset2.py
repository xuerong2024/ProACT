from collections import Counter
import pandas as pd
import shutil
import os
import random


# 定义路径
csvpath = "/disk1/wjr/dataset/NIHorg_dataset/Data_Entry_2017.csv"
imgpath = '/disk1/wjr/dataset/NIHorg_dataset/images'
newpath = '/disk1/wjr/dataset/NIHorg_dataset/newlocimg'

# 检查并创建新路径
os.makedirs(newpath, exist_ok=True)

# 读取 CSV 文件
csv = pd.read_csv(csvpath)

indexes = []
# 过滤出所有 Finding Label 为 'No Finding' 的行
filtered_lines = csv[csv["Finding Labels"]=='No Finding']
for ii in range(400, 800):
    # 使用布尔索引筛选行
    filtered_line = filtered_lines.iloc[ii]  # 获取当前行
    aa=filtered_line['Image Index'].split('_')[0]
    filtered_line_index = csv[csv["Image Index"].str.contains(aa)]
    if filtered_line_index.shape[0] == 1:
        indexes.append(filtered_line_index["Image Index"].values[0])
        print(filtered_line_index["Image Index"].values[0])

# 随机抽取 110 个路径
num_samples = 110
if len(indexes) < num_samples:
    print(f"可用的索引少于 {num_samples}，将抽取所有可用索引。")
    num_samples = len(indexes)

random_indexes = random.sample(indexes, num_samples)

# 迁移文件到新路径
for img_index in random_indexes:
    img_source_path = os.path.join(imgpath, img_index)
    img_destination_path = os.path.join(newpath, img_index)

    # 复制文件
    shutil.copyfile(img_source_path, img_destination_path)

# 输出结果
print(f"已成功迁移 {num_samples} 个图像到 {newpath}.")