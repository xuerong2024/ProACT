# '''计算山西dcm患病胸片每一期的数量'''
# import pandas as pd
# excel_path='G:\datasets\chenfei/nejm_shanxi_dataset\900txt_details/3centersdetail/subregions_label_shanxi_dcm_all.xlsx'
# txt_path='G:\datasets\chenfei/nejm_shanxi_dataset\900txt_details/3centersdetail/all.txt'
# shanxi_dcm_sick_95txt_path='G:\datasets\chenfei/nejm_shanxi_dataset\900txt_details/3centersdetail/shanxi_dcm_allsick_1807.txt'
# csv = pd.read_excel(excel_path)
# with open(txt_path, 'r', encoding='gbk') as file:
#     lines = file.readlines()
#
# stage1_num=0
# stage2_num=0
# stage3_num=0
# all_num=0
# for line in lines:
#     imgname=line.strip()
#     csv_line = csv.loc[(csv["胸片名称"] == imgname)]
#     if csv_line.size != 0:
#         if 'Health' in csv_line['胸片名称'].values[0]:
#             continue
#         with open(shanxi_dcm_sick_95txt_path, 'a') as f:
#             f.write(imgname+'\n')
#         all_num+=1
#         if 'Sick_Stage1' in csv_line['胸片名称'].values[0]:
#             stage1_num+=1
#         elif 'Sick_Stage2' in csv_line['胸片名称'].values[0]:
#             stage2_num+=1
#         elif 'Sick_Stage3' in csv_line['胸片名称'].values[0]:
#             stage3_num+=1
#
# print(f"all_num: {all_num}, stage1_num: {stage1_num}, stage2_num: {stage2_num}, stage3_num: {stage3_num}")

'''计算每个期别数量'''
import pandas as pd
excel_path='/disk3/wjr/dataset/nejm/shanxidataset/subregions_label_shanxi_all.xlsx'
txt_path='/disk3/wjr/dataset/nejm/shanxidataset/chinese_ilo_book.txt'
# shanxi_dcm_sick_95txt_path='G:\datasets\chenfei/nejm_shanxi_dataset\900txt_details/3centersdetail/shanxi_wsub_sick_95_185_8.txt'
csv = pd.read_excel(excel_path)
with open(txt_path, 'r', encoding='gbk') as file:
    lines = file.readlines()

stage1_num=0
stage2_num=0
stage3_num=0
all_num=0
names=[]
# 定义要提取的肺区列名
lung_zones = ['左上', '左中', '左下', '右上', '右中', '右下']

for line in lines:
    imgname = line.strip()
    csv_line = csv[csv["胸片名称"] == imgname]

    if csv_line.empty:
        continue  # 跳过未匹配的图像名

    # 遍历每个肺区列
    for zone in lung_zones:
        if zone in csv_line.columns:
            value = csv_line[zone].values[0]  # 取该列第一个值
            if pd.notna(value):  # 排除 NaN
                names.append(value)
# 去重并初始化计数字典
unique_names = set(names)
count_dict = {f"{name}_num": 0 for name in unique_names}

# 第二次遍历：统计频次
for line in lines:
    imgname = line.strip()
    csv_line = csv[csv["胸片名称"] == imgname]

    if csv_line.empty:
        continue

    for zone in lung_zones:
        if zone in csv_line.columns:
            value = csv_line[zone].values[0]
            if pd.notna(value):
                key = f"{value}_num"
                count_dict[key] += 1

# 打印结果
print(count_dict)
