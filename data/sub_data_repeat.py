from collections import Counter
from collections import defaultdict
txt_path='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_sick_biaozhun_subregions.txt'
# 读取文件并提取冒号后的部分
combinations = []
groups = defaultdict(list)
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if ':' in line:
            combo = line.split(':', 1)[1].split(',')[0]  # 取第一个冒号之后的内容
            if combo!='1/1' and combo!='1/0' and combo!='0/1' and combo!='0/0':
                combo='1/+'
            # combo = line.split(':', 1)[1]  # 取第一个冒号之后的内容
            combinations.append(combo)
            groups[combo].append(line)  # 保存整行

# 统计每种组合的出现次数
counts = Counter(combinations)
# 打印结果
for combo, count in counts.most_common():
    print(f"{combo}: {count}")

# Step 2: 找到最大频次
max_count = max(len(lines) for lines in groups.values())
print(f"Max frequency: {max_count}")

import random
# Step 3: 对每个组合进行过采样（随机重复），使其达到 max_count
balanced_lines = []
for combo, lines in groups.items():
    current_count = len(lines)
    if current_count < max_count:
        # 随机有放回采样，补足到 max_count
        sampled = random.choices(lines, k=max_count - current_count)
        balanced_lines.extend(lines + sampled)
    else:
        balanced_lines.extend(lines)

# 可选：打乱顺序
random.shuffle(balanced_lines)

# Step 4: 保存或使用 balanced_lines
output_path = txt_path.replace('.txt', '_balanced.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for line in balanced_lines:
        f.write(line + '\n')

print(f"Balanced dataset saved to: {output_path}")
print("Final counts per combo:")
for combo in sorted(groups.keys()):
    count = len([line for line in balanced_lines if line.split(':', 1)[1].split(',')[0] == combo])
    print(f"  {combo}: {count}")