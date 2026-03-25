from PIL import Image
import os


def downsample_images(source_folder, target_folder, new_size=(64, 64)):
    """
    使用双三次插值法下采样文件夹中的所有图片到指定大小，并保存到另一个文件夹。
    :param source_folder: 源图片文件夹路径
    :param target_folder: 目标文件夹路径，用于存放下采样后的图片
    :param new_size: 下采样大小，格式为(width, height)
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(source_folder, filename)
            img = Image.open(file_path)

            # 使用双三次插值法下采样图片
            img_downsampled = img.resize(new_size, Image.BICUBIC)

            # 保存到目标文件夹
            img_downsampled.save(os.path.join(target_folder, filename))


# 使用示例
source_folder = '/tmp/pycharm_project_ATD2/datasets/AID_testHR'  # 替换为源文件夹路径
target_folder = '/tmp/pycharm_project_ATD2/datasets/AID_test300'  # 替换为目标文件夹路径
downsample_images(source_folder, target_folder, new_size=(300,300))





































































































































