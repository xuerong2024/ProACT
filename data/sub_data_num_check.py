from collections import Counter
# txt_path='/disk3/wjr/dataset/nejm/guizhoudataset/guizhou_one_sick_subregions.txt'
txt_path='/disk3/wjr/dataset/nejm/shanxidataset/chinese_ilo_book_subregions.txt'

# 读取文件并提取冒号后的部分

combinations = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if ':' in line:
            combo = line.split(':', 1)[1].split(',')[0]  # 取第一个冒号之后的内容
            # combo = line.split(':', 1)[1]  # 取第一个冒号之后的内容
            combinations.append(combo)

# 统计每种组合的出现次数
counts = Counter(combinations)

# 打印结果
for combo, count in counts.most_common():
    print(f"{combo}: {count}")