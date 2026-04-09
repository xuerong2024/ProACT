import os
import random
root_dir='/disk3/wjr/dataset/CPNXray'
train_ratio = 0.6
val_ratio = 0.1
test_ratio = 0.3
# 创建保存文件名的文件
train_file = open('/disk3/wjr/dataset/CPNXray/train.txt', 'w')
val_file = open('/disk3/wjr/dataset/CPNXray/val.txt', 'w')
test_file = open('/disk3/wjr/dataset/CPNXray/test.txt', 'w')
train_num=0
val_num=0
test_num=0
# 配置
categories = ['COVID', 'NORMAL', 'PNEUMONIA']  # 类别文件夹名称
for class_index, category in enumerate(categories):
    category_path = os.path.join(root_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # 保存训练集文件名
    for image in train_images:
        train_file.write(f"{category}/{image}\t{class_index}\n")
        # train_file.write(f"{image}\n")
        train_num+=1

    # 保存验证集文件名
    for image in val_images:
        val_file.write(f"{category}/{image}\t{class_index}\n")
        val_num+=1

    # 保存测试集文件名
    for image in test_images:
        test_file.write(f"{category}/{image}\t{class_index}\n")
        test_num+=1

# 关闭文件
train_file.close()
val_file.close()
test_file.close()

print("文件名已保存到 train.txt, val.txt 和 test.txt")
print("train_num:", train_num)
print("val_num:", val_num)
print("test_num:", test_num)