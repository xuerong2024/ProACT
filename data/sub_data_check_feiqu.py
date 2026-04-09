from collections import defaultdict, Counter
# txt_path='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one_sick_subregions.txt'
# txt_path='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick_biaozhun_subregions_balanced_by_region_combo_weighted_yinying.txt'
txt_path='/disk3/wjr/dataset/nejm/shanxidataset/chinese_ilo_book_subregions.txt'
# 读取文件并提取冒号后的部分


from collections import defaultdict, Counter

regions = ['left_top', 'right_top', 'left_center', 'right_center', 'left_bottom', 'right_bottom']


combinations=defaultdict(list)

# 用于收集所有出现的 yinying 类型
observed_yinying = set()
observed_region = set()

with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

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
        if combo!='0/0' and combo!='0/1' and combo!='1/0' and combo!='1/1':
            combo='1+'

        observed_region.add(region_name)
        # 存储 combo
        combinations[region_name].append(combo)


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
