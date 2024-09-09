# main2更新：
# 获取后时相影像文件夹文件列表后，先筛选出img和tif文件（根据需求改动）
# 然后遍历

# -*- coding:utf-8 -*-
import os
from OpencvWork import *
from ShiftDetection import *
import csv


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
    img_ls[i] = img_ls[i].lower()                               # 变为小写

# 正确格式的后时相影像
corrected_files_ls = []

# 遍历后时相影像文件夹
for dir, dirpath, files in os.walk(arcpy.env.workspace):

    corrected_files_ls = files.copy()
    for file in files:
        if not (file.endswith(".img") or file.endswith(".tif")):
            corrected_files_ls.remove(file)

for i in range(0, len(corrected_files_ls)):
    temp = corrected_files_ls[i].split('.')
    file = temp[0]
    corrected_files_ls[i] = file.lower()

# 找出文件夹有的和确实的影像
common_file = list(set(img_ls).intersection(corrected_files_ls))
diff_file = list(set(img_ls).difference(corrected_files_ls))


# 遍历可以使用的影像



# 关闭文件
csv_file.close()
list_file.close()