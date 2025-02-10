
import os

def rename_files(folder_a, folder_b):
    # 获取文件夹A中的文件名列表
    filenames_a = os.listdir(folder_a)

    # 获取文件夹B中的文件名列表
    filenames_b = os.listdir(folder_b)

    # 确保两个文件夹中的文件数量相同
    assert len(filenames_a) == len(filenames_b), "Number of files in folders A and B do not match"

    # 遍历文件夹B中的文件，并将其重命名为与文件夹A中相同名称的文件
    for filename_a, filename_b in zip(filenames_a, filenames_b):
        src_path = os.path.join(folder_b, filename_b)
        dst_path = os.path.join(folder_b, filename_a)
        os.rename(src_path, dst_path)
        print(f"Renamed {filename_b} to {filename_a}")

# Example usage
folder_a = r"A:\B\ControlNet-main\dataset\onlyclothe\sketch"
folder_b = r"A:\B\ControlNet-main\dataset\onlyclothe\lineart_coarse"

rename_files(folder_a, folder_b)
#########跳过
# import os
#
#
# def change_file_extension(folder_path, old_extension, new_extension):
#     # 获取文件夹中所有文件名
#     filenames = os.listdir(folder_path)
#
#     # 遍历文件夹中的每个文件
#     for filename in filenames:
#         # 检查文件的后缀是否是指定的旧后缀
#         if filename.endswith(old_extension):
#             # 构造旧文件路径和新文件路径
#             old_file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, filename[:-len(old_extension)] + new_extension)
#
#             # 如果新文件路径已经存在，则跳过重命名操作
#             if os.path.exists(new_file_path):
#                 print(f"Skipping: {new_file_path} already exists.")
#             else:
#                 # 将文件重命名为新后缀
#                 os.rename(old_file_path, new_file_path)
#                 print(f"Renamed: {old_file_path} -> {new_file_path}")
#
#
# # 用法示例
# folder_path = r"A:\B\ControlNet-main\dataset\test\zjf (2)\zjf"
# new_extension = ".jpg"
# old_extension = ".png"
#
# change_file_extension(folder_path, old_extension, new_extension)
###########覆盖
import os
import shutil
#
# def change_file_extension(folder_path, old_extension, new_extension):
#     # 获取文件夹中所有文件名
#     filenames = os.listdir(folder_path)
#
#     # 遍历文件夹中的每个文件
#     for filename in filenames:
#         # 检查文件的后缀是否是指定的旧后缀
#         if filename.endswith(old_extension):
#             # 构造旧文件路径和新文件路径
#             old_file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, filename[:-len(old_extension)] + new_extension)
#
#             # 使用 shutil.move 替代 os.rename，可以覆盖同名文件
#             shutil.move(old_file_path, new_file_path)
#             print(f"Renamed: {old_file_path} -> {new_file_path}")
#
# # 用法示例
# folder_path = r"A:\B\ControlNet-main\dataset\test\zjf"
# new_extension = ".jpg"
# old_extension = ".png"
#
# change_file_extension(folder_path, old_extension, new_extension)
#
# import os
import cv2
import numpy as np

# 文件夹路径
# mask_folder = "E:\数据集\ControlnetDataset\controlnet-OnlyClothe\edit\mask"
#
# # 获取文件夹中所有文件名
# mask_filenames = os.listdir(mask_folder)
#
# # 遍历每个文件名，并尝试加载图像
# for filename in mask_filenames:
#     mask_path = os.path.join(mask_folder, filename)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#     # 检查是否成功加载图像
#     if mask is None:
#         print(f"Failed to load mask image: {filename}")
#     else:
#         print(f"Mask image loaded successfully: {filename}")
