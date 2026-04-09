import pandas as pd
from PIL import Image
import os
import random
import numpy as np
csvpath='/disk3/wjr/dataset/nejm/shanxidataset/subregions_label_shanxi.xlsx'
txtpath='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train.txt'
imgpath = '/disk3/wjr/dataset/nejm/shanxidataset/seg_rec_mask_224'
savepath='/disk3/wjr/dataset/nejm/shanxidataset/tempsubmiximg3/'
with open(txtpath, 'r', encoding='gbk') as file:
    lines = file.readlines()
csv = pd.read_excel(csvpath)
results = []
for line in lines:
    imgname = line.split('\n')[0]
    labelname = imgname.split('_')[0]
    img_path = os.path.join(imgpath, imgname.split('.png')[0] + '.png')
    mask_image = Image.open(img_path)
    csv_line = csv.loc[(csv["胸片名称"] == imgname)]

    if csv_line.size != 0:
        mask_new_subimage_org = np.asarray(mask_image)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()

        img_size = mask_image.size[0]
        left_upper_index = csv_line.columns.get_loc('左上')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        left_top_label = values[0, 0]
        if '0/0' in left_top_label or '0/1' in left_top_label:
            aa=1
        else:
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
        height = ymax - ymin
        width = xmax - xmin


        left_upper_index = csv_line.columns.get_loc('右上')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        right_top_label = values[0, 0]
        if '0/0' in right_top_label or '0/1' in right_top_label:
            aa=1
        else:
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]
        height = ymax - ymin
        width = xmax - xmin



        # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
        # plt.show()
        left_upper_index = csv_line.columns.get_loc('左中')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        left_center_label = values[0, 0]
        if '0/0' in left_center_label or '0/1' in left_center_label:
            aa=1
        else:
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
        height = ymax - ymin
        width = xmax - xmin


        left_upper_index = csv_line.columns.get_loc('右中')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        right_center_label = values[0, 0]
        if '0/0' in right_center_label or '0/1' in right_center_label:
            aa=1
        else:
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

        height = ymax - ymin
        width = xmax - xmin

        left_upper_index = csv_line.columns.get_loc('左下')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        left_bottom_label = values[0, 0]
        if '0/0' in left_bottom_label or '0/1' in left_bottom_label:
            aa=1
        else:
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

        height = ymax - ymin
        width = xmax - xmin

        left_upper_index = csv_line.columns.get_loc('右下')
        next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
        values = csv_line[next_three_columns].values
        xmin = values[0, 1] / 1024 * img_size
        ymin = values[0, 2] / 1024 * img_size
        xmax = values[0, 3] / 1024 * img_size
        ymax = values[0, 4] / 1024 * img_size
        right_bottom_label = values[0, 0]
        if '0/0' in right_bottom_label or '0/1' in right_bottom_label:
            aa=1
        else:
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

        height = ymax - ymin
        width = xmax - xmin
        org_left_mask = mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
        org_right_mask = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop

        if '0/0' in left_top_label or '0/1' in left_top_label:
            aa=1
        else:
            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(lines) - 1)
                line2 = lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = csv.loc[(csv["胸片名称"] == imgname2)]
                if csv_line2.size == 0:
                    if 'Health' in imgname2:
                        left_top_label2 = '0/0'
                        left_center_label2 = '0/0'
                        left_bottom_label2 = '0/0'
                        right_top_label2 = '0/0'
                        right_center_label2 = '0/0'
                        right_bottom_label2 = '0/0'
                        choosen_index = 1
                        mixed_mask1 = org_left_mask
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        # 保存图像到文件
                        mixed_mask1.save(savepath + imgname.split('.png')[0] + '_left.png')  # 保存为 PNG 格式
                        mixed_mask2 = Image.fromarray(org_right_mask, mode='L')
                else:
                    # if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(imgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(img_path2)
                    img_size = mask_image2_2.size[0]
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('左上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0, 0]
                    if '0/0' in left_top_label2 or '0/1' in left_top_label2:
                        aa = 1
                    else:
                        mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                               int(ymin):int(ymax),
                                                                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label2 = values[0, 0]
                    if '0/0' in right_top_label2 or '0/1' in right_top_label2:
                        aa = 1
                    else:
                        mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                                int(ymin):int(ymax),
                                                                                                int(xmin):int(xmax)]

                    height = ymax - ymin
                    width = xmax - xmin

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label2 = values[0, 0]
                    if '0/0' in left_center_label2 or '0/1' in left_center_label2:
                        aa = 1
                    else:
                        mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label2 = values[0, 0]
                    if '0/0' in right_center_label2 or '0/1' in right_center_label2:
                        aa = 1
                    else:
                        mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label2 = values[0, 0]
                    if '0/0' in left_bottom_label2 or '0/1' in left_bottom_label2:
                        aa = 1
                    else:
                        mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label2 = values[0, 0]
                    if '0/0' in right_bottom_label2 or '0/1' in right_bottom_label2:
                        aa = 1
                    else:
                        mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')
                    mixed_mask1.save(savepath + imgname.split('.png')[0] + '_left.png')
            # 第一组数据
            results.append({
                "胸片名称": imgname + '_left',
                "左上": f"'{left_top_label}",
                "左中": f"'{left_center_label}",
                "左下": f"'{left_bottom_label}",
                "右上": f"'{right_top_label2}",
                "右中": f"'{right_center_label2}",
                "右下": f"'{right_bottom_label2}"
            })


        if '0/0' in right_top_label or '0/1' in right_top_label:
            aa=1
        else:
            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(lines) - 1)
                line2 = lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = csv.loc[(csv["胸片名称"] == imgname2)]
                if csv_line2.size == 0:
                    if 'Health' in imgname2:
                        left_top_label2 = '0/0'
                        left_center_label2 = '0/0'
                        left_bottom_label2 = '0/0'
                        right_top_label2 = '0/0'
                        right_center_label2 = '0/0'
                        right_bottom_label2 = '0/0'
                        choosen_index = 1
                        mixed_mask1 = org_left_mask
                        mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                        # 保存图像到文件  # 保存为 PNG 格式
                        mixed_mask2 = Image.fromarray(org_right_mask, mode='L')
                        # 保存图像到文件
                        mixed_mask2.save(savepath + imgname.split('.png')[0] + '_right.png')
                else:
                    # if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(imgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(img_path2)
                    img_size = mask_image2_2.size[0]
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('左上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0, 0]
                    if '0/0' in left_top_label2 or '0/1' in left_top_label2:
                        aa = 1
                    else:
                        mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                               int(ymin):int(ymax),
                                                                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label2 = values[0, 0]
                    if '0/0' in right_top_label2 or '0/1' in right_top_label2:
                        aa = 1
                    else:
                        mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                                int(ymin):int(ymax),
                                                                                                int(xmin):int(xmax)]

                    height = ymax - ymin
                    width = xmax - xmin

                    # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
                    # plt.show()
                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label2 = values[0, 0]
                    if '0/0' in left_center_label2 or '0/1' in left_center_label2:
                        aa = 1
                    else:
                        mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label2 = values[0, 0]
                    if '0/0' in right_center_label2 or '0/1' in right_center_label2:
                        aa = 1
                    else:
                        mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label2 = values[0, 0]
                    if '0/0' in left_bottom_label2 or '0/1' in left_bottom_label2:
                        aa = 1
                    else:
                        mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label2 = values[0, 0]
                    if '0/0' in right_bottom_label2 or '0/1' in right_bottom_label2:
                        aa = 1
                    else:
                        mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                        int(xmin):int(xmax)] = mask_new_subimage_org2[
                                               int(ymin):int(ymax),
                                               int(xmin):int(xmax)]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')
                    mixed_mask2.save(savepath + imgname.split('.png')[0] + '_right.png')


            # 第二组数据
            results.append({
                "胸片名称": imgname + '_right',
                "左上": f"'{left_top_label2}",
                "左中": f"'{left_center_label2}",
                "左下": f"'{left_bottom_label2}",
                "右上": f"'{right_top_label}",
                "右中": f"'{right_center_label}",
                "右下": f"'{right_bottom_label}"
            })

# 转换为 DataFrame
df = pd.DataFrame(results)

# 保存为 CSV 文件
df.to_csv(savepath+"sub_mix_results.csv", index=False, encoding="utf-8")