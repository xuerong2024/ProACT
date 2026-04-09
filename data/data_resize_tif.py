import pathlib
import openslide
import numpy as np
from PIL import Image
folder = 'I:\shanxisaomiao\huanbing\org_img'
# aa=os.walk(folder)
data_root = pathlib.Path(folder)
all_files = list(data_root.glob('*.tif'))
all_files = [str(path) for path in all_files]


def resize_wsi_image(input_path, output_path, max_size=(6000, 6000)):
    """
    使用 OpenSlide 快速缩放 WSI 图像。

    :param input_path: 输入 WSI 图像的路径
    :param output_path: 输出图像的路径
    :param max_size: 图像的最大尺寸 (宽, 高)，默认为 (6000, 6000)
    """
    try:
        # 打开 WSI 图像
        slide = openslide.OpenSlide(input_path)

        # 获取图像尺寸
        original_width, original_height = slide.dimensions
        print(f"Original size: {original_width}x{original_height}")

        # 计算新的尺寸，保持纵横比
        if original_width > max_size[0] or original_height > max_size[1]:
            ratio = min(max_size[0] / original_width, max_size[1] / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            print(f"Resizing to: {new_width}x{new_height}")

            # 读取缩放后的图像区域
            region = slide.read_region((0, 0), 0, (new_width, new_height))
            img = region.convert('RGB')  # 将图像转换为 RGB 格式

            # 保存缩小后的图像
            img.save(output_path, optimize=True, quality=95)
            print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")


# 示例用法
input_path = 'I:\shanxisaomiao\huanbing/org_img/guan-di_87_zhang-xiao_20140802.tif'
output_path = 'I:\shanxisaomiao\huanbing/temp\guan-di_87_zhang-xiao_20140802.tif'
resize_wsi_image(input_path, output_path, max_size=(6000, 6000))


# # 设置新的最大像素限制（例如 10 亿像素）
# # Image.MAX_IMAGE_PIXELS = 1000000000
# img = Image.open('I:\shanxisaomiao\huanbing/org_img/guan-di_87_zhang-xiao_20140802.tif')
# max_size = (6000, 6000)  # 设置最大尺寸
# img.thumbnail(max_size)
# # 保存缩小后的图像
# img.save('I:\shanxisaomiao\huanbing/temp\guan-di_87_zhang-xiao_20140802.tif')
# a=1
# # for impath in all_files:
# #     img = Image.open('I:\shanxisaomiao\huanbing/org_img/guan-di_74_jia-sheng-tong_197811.tif')
# #     max_size = (6000, 6000)  # 设置最大尺寸
# #     img.thumbnail(max_size)
# #     # 保存缩小后的图像
# #     img.save('I:\shanxisaomiao\huanbing/temp\guan-di_74_jia-sheng-tong_197811.tif')
# #     a=1