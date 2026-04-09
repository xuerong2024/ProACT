import random
from collections import defaultdict, Counter

# 设置随机种子以保证可复现
random.seed(42)

txt_path = '/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt'
output_path = '/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions_balanced.txt'

regions = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']

# 存储每行的 (region, combo, full_line)
data_by_region = defaultdict(list)  # key: region, value: list of (combo, line)

# 第一步：读取并解析所有有效行
with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    stripped = line.strip()
    if not stripped:
        continue
    if ':' not in stripped:
        continue

    region_part, content = stripped.split(':', 1)
    region_name = None
    for region in regions:
        if region in region_part:
            region_name = region
            break
    if region_name is None:
        continue  # 跳过无法识别 region 的行

    fields = [f.strip() for f in content.split(',')]
    combo = fields[0] if fields else 'unknown'

    # 映射 combo
    if combo not in {'0/0', '0/1', '1/0', '1/1'}:
        combo = '1+'

    data_by_region[region_name].append((combo, line))  # 保留原始行（含换行符）

# 第二步：对每个 region 做均衡采样
balanced_lines = []

for region, combo_line_list in data_by_region.items():
    # 统计 combo 分布
    combo_counts = Counter([combo for combo, _ in combo_line_list])
    if not combo_counts:
        continue

    min_count = min(combo_counts.values())
    print(f"Region {region}: min class count = {min_count}")

    # 按 combo 分组
    grouped = defaultdict(list)
    for combo, line in combo_line_list:
        grouped[combo].append(line)

    # 对每个 combo 随机采样 min_count 个
    for combo, lines_list in grouped.items():
        sampled = random.sample(lines_list, min_count)  # 无放回随机采样
        balanced_lines.extend(sampled)

# 可选：打乱最终顺序（避免 region 集中）
# random.shuffle(balanced_lines)

# 第三步：写入新文件
with open(output_path, 'w', encoding='utf-8') as f_out:
    f_out.writelines(balanced_lines)

print(f"\n✅ Balanced file saved to: {output_path}")
print(f"Total lines after balancing: {len(balanced_lines)}")