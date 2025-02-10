# from PIL import Image
#
#
# def invert_colors(input_path, output_path):
#     # 打开图像
#     image = Image.open(input_path)
#
#     # 获取图像的宽度和高度
#     width, height = image.size
#
#     # 创建一个新的图像对象
#     inverted_image = Image.new('RGB', (width, height))
#
#     # 循环遍历每个像素并反色
#     for x in range(width):
#         for y in range(height):
#             # 获取原始图像的像素值
#             original_pixel = image.getpixel((x, y))
#
#             # 反色操作
#             inverted_pixel = tuple(255 - value for value in original_pixel)
#
#             # 在新图像中设置反色后的像素值
#             inverted_image.putpixel((x, y), inverted_pixel)
#
#     # 保存反色后的图像
#     inverted_image.save(output_path)
#
#
# # 输入和输出文件路径
# input_image_path = r'C:\Users\26272\Desktop\1.png'  # 请替换为实际的输入图像文件路径
# output_image_path = 'output_image.jpg'  # 请替换为实际的输出图像文件路径
#
# 执行反色操作
# invert_colors(input_image_path, output_image_path)
from PIL import Image
import os
import time


def invert_colors(input_folder, output_folder):
    start_time = time.time()  # Record the start time

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建输入文件的完整路径
        input_path = os.path.join(input_folder, filename)

        # 打开图像
        image = Image.open(input_path)

        # 获取图像的宽度和高度
        width, height = image.size

        # 创建一个新的图像对象
        inverted_image = Image.new('RGB', (width, height))

        # 循环遍历每个像素并反色
        for x in range(width):
            for y in range(height):
                # 获取原始图像的像素值
                original_pixel = image.getpixel((x, y))

                # 反色操作
                inverted_pixel = tuple(255 - value for value in original_pixel)

                # 在新图像中设置反色后的像素值
                inverted_image.putpixel((x, y), inverted_pixel)

        # 构建输出文件的完整路径
        output_path = os.path.join(output_folder, filename)

        # 保存反色后的图像
        inverted_image.save(output_path)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")
    savetimefile="InvertSketchImageTime.txt"
    with open(savetimefile,'w') as file:
        file.write(f"start_time:{start_time}\n")
        file.write(f"end_time:{end_time}\n")
        file.write(f"elapsed_time:{elapsed_time}\n")
    print("变量已保存到文件",savetimefile)

# 输入和输出文件夹路径
input_folder_path = r'/SYJ/Anjou/TControlNet/onlyclothe/lineart_coarse'  # 请替换为实际的输入文件夹路径
output_folder_path = r'/SYJ/Anjou/TControlNet/onlyclothe/source'  # 请替换为实际的输出文件夹路径

# 执行反色操作
invert_colors(input_folder_path, output_folder_path)

# from PIL import Image
# import numpy as np
#
# from PIL import Image
# import numpy as np
#
# def invert_colors(image_path, save_path):
#     # Open the image
#     img = Image.open(image_path)
#
#     # Convert the image to RGB mode if it's in RGBA mode
#     if img.mode == 'RGBA':
#         img = img.convert('RGB')
#
#     # Convert the image to a NumPy array
#     img_array = np.array(img)
#
#     # Invert the colors (black to white, white to black)
#     inverted_img_array = 255 - img_array
#
#     # Create a new Image from the inverted NumPy array
#     inverted_img = Image.fromarray(inverted_img_array)
#
#     # Save the result
#     inverted_img.save(save_path)
#     print(f"Image with inverted colors saved at: {save_path}")
#
# # Replace 'input_image.jpg' and 'output_image.jpg' with your actual file paths
# input_image_path = r'C:\Users\26272\Desktop\mask.png'
# output_image_path = 'output_image.jpg'
#
# invert_colors(input_image_path, output_image_path)

