import os

txtpath = '/disk3/wjr/dataset/nejm/shanxidataset/sick_health_5fold_txt/sick_subregion.txt'
if not os.path.exists(txtpath):
    raise FileNotFoundError(f"文件 {txtpath} 不存在！")

with open(txtpath, "r") as f:
    contents = f.read().splitlines()

for ii in range(1):  # 调整循环次数
    txt_path = '/disk3/wjr/dataset/nejm/shanxidataset/sick_health_5fold_txt/fold1_train.txt'
    new_txt_path = '/disk3/wjr/dataset/nejm/shanxidataset/sick_health_5fold_txt/fold1_train_sick_subregion.txt'

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"训练文件 {txt_path} 不存在！")

    with open(txt_path, "r") as f:
        contents_train = f.read().splitlines()

    with open(new_txt_path, "w") as f:
        for line in contents_train:
            if line in contents:
                f.write(line + "\n")

    print(f"完成第 {ii+1} 轮处理，结果保存至：{new_txt_path}")