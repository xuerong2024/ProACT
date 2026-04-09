import re


def filter_and_save_lines(input_path, output_path):
    # 定义要匹配的模式（包括 none 和 x/y 形式中的 p/q 组合）
    patterns = ['p/p', 'p/q', 'q/p', 'q/q', 'none']

    filtered_lines = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 检查行末尾是否以 ",xxx" 结尾，且 xxx 属于目标模式
            # 例如: ...:1/2,q/q  → 我们关心逗号后的部分
            parts = line.split(',')
            if len(parts) >= 2:
                suffix = parts[-1]  # 取最后一个逗号之后的内容
                if suffix in patterns:
                    filtered_lines.append(line)

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line + '\n')

    print(f"已筛选 {len(filtered_lines)} 行，保存至 {output_path}")


# 使用示例
if __name__ == "__main__":
    input_file = "/disk3/wjr/dataset/nejm/shanxidataset/chinese_ilo_book_subregions.txt"  # 替换为你的输入文件路径
    output_file = "/disk3/wjr/dataset/nejm/shanxidataset/chinese_ilo_book_subregions_pq.txt"  # 输出文件名
    filter_and_save_lines(input_file, output_file)