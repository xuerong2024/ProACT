import random
from collections import defaultdict, Counter

txt_path = '/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick_biaozhun_subregions.txt'

# --- 1. 读取文件并解析数据 ---
# 数据结构: data_store[region][combo][yinying_type] = [list of full lines]
data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue

        try:
            # --- 提取 region ---
            region_part = line.split(':', 1)[0]
            region_parts = region_part.split('_')
            if len(region_parts) >= 1:
                region = region_parts[-1]  # 取最后一个下划线后的部分作为 region
            else:
                continue

            # --- 提取 combo 和 yinying_type ---
            content = line.split(':', 1)[1]
            fields = [f.strip() for f in content.split(',')]

            if len(fields) < 2:
                continue

            raw_combo = fields[0]
            # --- 标准化 combo ---
            if raw_combo in ['1/1', '1/0', '0/1', '0/0']:
                combo = raw_combo
            else:
                combo = '1/+'  # 将其他组合统一归类

            yinying_type_field = fields[-1]  # 假设 yinying_type 是最后一个字段
            yinying_type = yinying_type_field.split('/')[0]  # 取 '/' 前的部分作为 yinying_type

            # --- 存储整行 ---
            data_store[region][combo][yinying_type].append(line)

        except Exception as e:
            print(f"Warning: Skipping malformed line: {line}. Error: {e}")
            continue

# --- 2. 在每个 region 内，按 combo 进行分层过采样 ---
balanced_lines = []

# 遍历每个 region
for region, combo_data in data_store.items():
    print(f"\n--- Balancing Region: {region} ---")

    # 1. 计算该 region 内每个 combo 的总样本数
    combo_totals = {combo: sum(len(lines) for lines in ying_data.values())
                    for combo, ying_data in combo_data.items()}

    if not combo_totals:
        continue

    # 2. 确定该 region 的目标数量 (通常是最大 combo 数量)
    target_count = max(combo_totals.values())
    print(f"  Target count for all combos in {region}: {target_count}")

    # 3. 对该 region 内的每个 combo 进行过采样
    for combo, ying_data in combo_data.items():
        current_total = combo_totals[combo]
        print(f"  Processing Combo: {combo} (Current: {current_total})")

        if current_total >= target_count:
            # 数量已够或超过，直接全部加入
            for lines in ying_data.values():  # 遍历所有 yinying_type 下的行
                balanced_lines.extend(lines)
            print(f"    -> No sampling needed. Added {current_total} lines.")
        else:
            # 需要过采样
            samples_needed = target_count - current_total
            print(f"    -> Need to sample {samples_needed} more lines.")

            # --- 修正后的采样逻辑 ---

            # 收集该 combo 下的所有行，并准备用于【反比】加权采样的权重
            all_lines_for_this_combo = []
            inverse_weights_for_sampling = []  # 注意这里是反比权重

            # 遍历该 combo 下的每个 yinying_type
            for yinying_type, lines in ying_data.items():
                count = len(lines)
                if count > 0:
                    all_lines_for_this_combo.extend(lines)
                    # 为这些行设置【反比】权重，权重 = 1 / 该 yinying_type 在此 combo 中的样本数
                    # 这样可以保证采样时，原始 yinying_type 比例低的样本被选中的概率更大
                    # 从而在采样后使得各 yinying_type 数量趋于一致
                    inverse_weights_for_sampling.extend([1.0 / count] * count)

            if not all_lines_for_this_combo:
                print(f"    Warning: No lines found for combo '{combo}' in region '{region}' to sample from!")
                # 将现有的行也加入，即使数量不足
                for lines in ying_data.values():
                    balanced_lines.extend(lines)
                continue  # 避免除零错误或空列表采样

            # 执行【反比权重】随机采样
            try:
                sampled_lines = random.choices(
                    all_lines_for_this_combo,
                    weights=inverse_weights_for_sampling,
                    k=samples_needed
                )
                # 将原始行和采样行都加入结果
                # 原始行
                for lines in ying_data.values():
                    balanced_lines.extend(lines)
                # 采样行
                balanced_lines.extend(sampled_lines)
                print(f"    -> Successfully sampled {len(sampled_lines)} lines (INVERSE weighted by yinying_type).")

            except Exception as e:
                print(f"    Error during INVERSE weighted sampling for {region}-{combo}: {e}")
                # Fallback: 如果加权采样出错（例如所有权重都是0，虽然不太可能），则进行无权重采样
                try:
                    sampled_lines = random.choices(all_lines_for_this_combo, k=samples_needed)
                    for lines in ying_data.values():
                        balanced_lines.extend(lines)
                    balanced_lines.extend(sampled_lines)
                    print(f"    -> Fallback: Sampled {len(sampled_lines)} lines (UNWEIGHTED).")
                except Exception as e2:
                    print(f"    -> Fallback failed: {e2}")
                    # 如果连无权重采样都失败（例如 samples_needed > available lines），则只加原始行
                    for lines in ying_data.values():
                        balanced_lines.extend(lines)

# --- 3. 打乱顺序并保存 ---
random.shuffle(balanced_lines)

output_path = txt_path.replace('.txt', '_balanced_by_region_combo_weighted_yinying.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for line in balanced_lines:
        f.write(line + '\n')

print(f"\nBalanced dataset saved to: {output_path}")

# --- 4. (可选) 验证最终各组的平衡性 ---
# 重新统计以验证结果
verification_store = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
with open(output_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue
        try:
            region_part = line.split(':', 1)[0]
            region_parts = region_part.split('_')
            if len(region_parts) >= 1:
                region = region_parts[-1]
            else:
                continue

            content = line.split(':', 1)[1]
            fields = [f.strip() for f in content.split(',')]
            if len(fields) < 2:
                continue

            raw_combo = fields[0]
            if raw_combo in ['1/1', '1/0', '0/1', '0/0']:
                combo = raw_combo
            else:
                combo = '1/+'

            yinying_type_field = fields[-1]
            yinying_type = yinying_type_field.split('/')[0]

            verification_store[region][combo][yinying_type] += 1
        except Exception as e:
            # print(f"Warning during verification: {e}")
            continue

print("\n--- Final Verification ---")
for region in sorted(verification_store.keys()):
    print(f"\n--- Region: {region} ---")
    region_combo_totals = defaultdict(int)
    region_combo_yinying_details = defaultdict(lambda: defaultdict(int))

    for combo in sorted(verification_store[region].keys()):
        total_for_combo = sum(count for count in verification_store[region][combo].values())
        region_combo_totals[combo] = total_for_combo
        print(f"  Combo {combo}: Total = {total_for_combo}")
        for yinying_type, count in verification_store[region][combo].items():
            print(f"    Yinying '{yinying_type}': {count}")

    # 检查该 region 内 combo 数量是否平衡
    counts_list = list(region_combo_totals.values())
    if counts_list and max(counts_list) == min(counts_list):
        print(f"  Status: COMBO counts are balanced within this region.")
    else:
        print(f"  Status: COMBO counts are NOT balanced. Min: {min(counts_list)}, Max: {max(counts_list)}")
