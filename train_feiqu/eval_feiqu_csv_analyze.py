import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/disk3/wjr/workspace/sec_proj4/baseline_expeript/feiqu5cls_2global/224/contra_20251127/wmask_subcls_nodropout/convnext_databig_allsick_health_addativepooling/224_batch60_lr5e-05_testall/guizhou_test_local_global_result.csv')


# 1. 统计错位的样本数量：局部预测错误（即某些肺区错误），全局预测正确
def check_local_and_global(row):
    # 判断每个肺区的预测是否正确
    local_errors = (
            (row['left_top_label'] != row['left_top_pred']) |
            (row['right_top_label'] != row['right_top_pred']) |
            (row['left_center_label'] != row['left_center_pred']) |
            (row['right_center_label'] != row['right_center_pred']) |
            (row['left_bottom_label'] != row['left_bottom_pred']) |
            (row['right_bottom_label'] != row['right_bottom_pred'])
    )
    # 如果有局部错误，且全局预测正确
    return local_errors and (row['Finding Labels'] == row['pred'])


# 筛选出局部预测错误且全局预测正确的样本
incorrect_local_correct_global = df[df.apply(check_local_and_global, axis=1)]


# # 2. 统计每个肺区的错误类型和错误预测
# def analyze_lung_zone_errors(row):
#     zone_errors = {zone: {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0} for zone in
#                    ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']}
#     error_types = {'0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                    '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                    '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                    '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                    '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}}
#
#     lung_zones = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']
#
#     for lung_zone in lung_zones:
#         label = row[f'{lung_zone}_label']
#         pred = row[f'{lung_zone}_pred']
#
#         if label != pred:  # 如果预测错误
#             zone_errors[lung_zone][pred] += 1
#             error_types[label][pred] += 1
#
#     return zone_errors, error_types


# 3. 对错位样本进行分析，统计每个肺区的错误类型
# error_details = df.apply(analyze_lung_zone_errors, axis=1)


# 定义错误类型统计字典
def get_error_statistics(df):
    error_types_by_lung = {
        'left_top': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        },
        'right_top': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        },
        'left_center': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        },
        'right_center': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        },
        'left_bottom': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        },
        'right_bottom': {
            '0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
            '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}
        }
    }

    # 遍历每一行数据进行错误类型统计
    for _, row in df.iterrows():
        for lung_area in ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']:
            label = row[f'{lung_area}_label']
            pred = row[f'{lung_area}_pred']

            if label != pred:  # 只统计预测错误的情况
                error_types_by_lung[lung_area][label][pred] += 1

    # 输出每个肺区的详细错误统计
    for lung_area in error_types_by_lung:
        print(f"{lung_area} 错误类型统计：")
        for label in error_types_by_lung[lung_area]:
            print(f"  {label}:")
            for pred in error_types_by_lung[lung_area][label]:
                count = error_types_by_lung[lung_area][label][pred]
                if count > 0:  # 只输出有错误的类别
                    print(f"    {label} 错误预测为 {pred}: {count} 次")
        print()


# 假设你的数据已经加载到df变量中
get_error_statistics(df)
#
# # 提取出每个肺区的错误统计数据
# total_zone_errors = {zone: {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0} for zone in
#                      ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']}
# total_error_types = {'0_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                      '0_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                      '1_0': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                      '1_1': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0},
#                      '1+': {'0_0': 0, '0_1': 0, '1_0': 0, '1_1': 0, '1+': 0}}
#
# # 累加错误数据
# for zone_errors, error_types in error_details:
#     for zone in total_zone_errors:
#         for error_type in total_zone_errors[zone]:
#             total_zone_errors[zone][error_type] += zone_errors[zone][error_type]
#     for label in total_error_types:
#         for error_type in total_error_types[label]:
#             total_error_types[label][error_type] += error_types[label][error_type]
#
# # 输出结果：全局正确，局部错误的统计
# print(f"错位样本数量 (局部错误，全局正确): {len(incorrect_local_correct_global)}")
#
# # 统计全局正确时，哪个肺区的错误最多
# max_zone_errors = {zone: max(errors.items(), key=lambda x: x[1]) for zone, errors in total_zone_errors.items()}
# print("\n每个肺区错误最多的类别：")
# for zone, (error_type, count) in max_zone_errors.items():
#     print(f"{zone}: {error_type} 错误 {count} 次")
#
# # 输出每个肺区的所有错误类型统计
# print("\n每个肺区的错误类型统计：")
# for zone, error_types in total_zone_errors.items():
#     print(f"\n{zone} 错误类型统计：")
#     for error_type, count in error_types.items():
#         print(f"  {error_type}: {count} 次")
#
# # 输出全局错误类型统计
# print("\n全局错误类型统计：")
# for label in total_error_types:
#     print(f"\n{label} 错误类型统计：")
#     for error_type, count in total_error_types[label].items():
#         print(f"  {error_type}: {count} 次")