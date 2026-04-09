from collections import defaultdict, Counter
# txt_path='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one_sick_subregions.txt'
# txt_path='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick_biaozhun_subregions_balanced_by_region_combo_weighted_yinying.txt'
txt_path='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt'
# 读取文件并提取冒号后的部分


from collections import defaultdict, Counter

# 定义合法的区域 (这部分通常是固定的，或者也可以动态提取)
# regions = ['lefttop', 'righttop', 'leftcenter', 'rightbottom', 'leftbottom', 'rightbottom']
regions = ['left_top', 'right_top', 'left_center', 'right_bottom', 'left_bottom', 'right_bottom']

# 创建嵌套的 defaultdict 结构
# combinations[yinying_type][region_name] = list of combos
# combinations = defaultdict(lambda: defaultdict(list))
combinations=defaultdict(list)

# 用于收集所有出现的 yinying 类型
observed_yinying = set()
observed_region = set()

with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 安全地提取 region_part
        # parts = line.split('_')
        # if len(parts) < 2:
        #     continue
        #
        # # 提取 region_part（例如 left_top）
        # # 假设 region_part 是最后两个下划线部分
        # region_part_raw = parts[-2] + '_' + parts[-1].split(':', 1)[0]
        # region_part_clean = region_part_raw.replace('_', '')  # 转为 lefttop 格式
        region_part_clean = line.split(':', 1)[0]
        for region in regions:
            if region in region_part_clean:
                region_name=region
                break


        if ':' not in line:
            continue

        # 提取冒号后的内容
        content = line.split(':', 1)[1]
        fields = [f.strip() for f in content.split(',')]
        # if len(fields) < 2:
        #     continue  # 至少要有 combo 和 yinying

        combo = fields[0]
        # yinying_type = fields[-1].split('/')[0]  # 最后一个是 yinying

        # 收集观察到的 yinying 类型
        # observed_yinying.add(yinying_type)
        observed_region.add(region_name)

        # 存储 combo
        combinations[region_name].append(combo)

# --- 输出结果 ---
# 将 observed_yinying 转换为排序列表以获得一致的输出顺序
# 你可以根据需要调整排序方式，例如按字母顺序 sorted(list(observed_yinying))
# sorted_yinying = sorted(list(observed_yinying))
sorted_region = sorted(list(observed_region))
for region in sorted_region:
    print(f"\n--- region: {region} ---")
    combo_list = combinations[region]
    if combo_list:
            counts = Counter(combo_list)
            for combo, count in counts.most_common():
                print(f" {region}:  {combo}: {count}")
    else:
            print(f"  {region}: (no data)")
