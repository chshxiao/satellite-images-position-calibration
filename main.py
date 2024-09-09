# -*- coding:utf-8 -*-
import os
from OpencvWork import *
from ShiftDetection import *
import csv
import shutil


def remove_dir_file(path):
    """
    删除文件夹内所有文件
    :param path:
    :return:
    """

    # 检查文件夹是否存在
    if not os.path.exists(path):
        return

    # 删除文件
    file_ls = os.listdir(path)
    for file in file_ls:
        os.remove(os.path.join(path, file))


# arcpy环境变量设置
arcpy.env.overwriteOutput = 1
arcpy.env.workspace = r"E:\课件\编程\Python\卫星影像对齐\影像"

# 输出csv所在文件夹
csv_path = r"E:\课件\编程\Python\卫星影像对齐"
csv_name = r"\结果.csv"

# 影像列表
list_path = r"E:\课件\编程\Python\卫星影像对齐"
list_filename = r"\影像列表.txt"

# 前时相影像
post_timestamp_path = r"E:\课件\编程\Python\卫星影像对齐\前时相影像"
base_img = post_timestamp_path + r"\MLC_shift.img"

# 打开csv和影像列表
try:
    csv_file = open(csv_path + csv_name, 'w')
    list_file = open(list_path + list_filename, 'r')
except Exception as e:
    print(str(e))
csv_writer = csv.writer(csv_file)

# 读取影像列表
img_ls = list_file.readlines()
for i in range(len(img_ls)):
    img_ls[i] = img_ls[i].replace("\n", "")
    img_ls[i] = img_ls[i].lower()


# 遍历后时相影像文件夹
for dir, dirpath, files in os.walk(arcpy.env.workspace):

    for file in files:

        # 可以用的影像
        lower_file = file.lower()

        if lower_file.endswith("c.img") or lower_file.endswith("pan.img") or lower_file.endswith("pan2.img") or \
                lower_file.endswith("pan1.img") or lower_file.endswith("rgb.img") or \
                (lower_file.startswith("gf1") and lower_file.endswith(".img") and not lower_file[-5:] in (
                "c.img", "p.img", "b.img", "m.img", "f.img")):
        # if True:

            lower_file = file.split('.')[0].lower()

            # 若影像在影像列表中，进行下一步
            if lower_file not in img_ls:
                continue


            # 影像属性
            img_path = arcpy.env.workspace + "\\" + file
            img_raster = arcpy.Raster(img_path)
            img_rect = str(img_raster.extent.XMin) + " " + \
                       str(img_raster.extent.YMin) + " " + \
                       str(img_raster.extent.XMax) + " " + \
                       str(img_raster.extent.YMax)
            cell_size = img_raster.meanCellWidth
            print("获取" + file + "栅格属性")


            # # 检查储存要素类gdb是否已存在
            # gdb_path = r"E:\课件\编程\Python\卫星影像对齐\测试.gdb"         # 存放gdb的路径
            # if not os.path.exists(gdb_path):
            #     # 创建gdb
            #     arcpy.CreateFileGDB_management(os.path.dirname(gdb_path), os.path.basename(gdb_path))


            # # 画出相同范围面要素类
            # arcpy.management.CreateFeatureclass(gdb_path, "影像边界", "POLYGON", '#', '#', '#',
            #                                     img_raster.extent.spatialReference)
            # arcpy.CopyFeatures_management(img_raster.extent.polygon, os.path.join(gdb_path, "影像边界"))
            # # 面要素类与基准图取并集


            # 储存裁剪后前时相影像文件夹
            clip_base_dir = post_timestamp_path + r"\裁剪后"
            if not os.path.exists(clip_base_dir):
                os.makedirs(clip_base_dir)


            # 裁剪前时相落图
            # clip_base_img = arcpy.env.workspace + r"\前时相落图\裁剪基准落图.img"
            # base_img = r"E:\课件\编程\Python\卫星影像对齐\前时相影像\translation_output.img"
            clip_base_img = clip_base_dir + r"\MLC_shift_clip.tif"
            # clip_base_img = r"E:\课件\编程\Python\卫星影像对齐\前时相影像\translation_output_clip.img"
            arcpy.Clip_management(base_img, img_rect, clip_base_img, '#', 0, '#', "MAINTAIN_EXTENT")
            print("裁剪前时相影像")


            # 对两图计算SIFT匹配点
            img1 = cv_imread(arcpy.env.workspace + "\\" + file)
            img2 = cv_imread(clip_base_img)
            print(file + "正在进行匹配")
            sift_result = split_image_and_sift(img1, img2)
            print(file + "完成匹配")


            # 判断影像是否偏移
            shift_result = detect_shift(sift_result, cell_size)
            print(file + "shift_result")


            # 结果输入csv
            csv_writer.writerow([lower_file, shift_result])
            print(file + "结果写入csv")
            print('\n\n')


            # 删除裁剪后前时相影像
            remove_dir_file(clip_base_dir)
            print('临时裁剪数据已删除')


# 关闭文件
csv_file.close()
list_file.close()
