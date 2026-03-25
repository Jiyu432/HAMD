import numpy as np
import cv2
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from tqdm import tqdm
def reduce_ringing_artifacts(image, kernel_size=3):
    """替代方案：双边滤波"""
    if len(image.shape) == 3:
        # 分别处理每个通道
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:,:,c] = cv2.bilateralFilter(
                image[:,:,c],
                d=kernel_size,
                sigmaColor=15,
                sigmaSpace=15
            )
        return filtered
    else:
        image_32f = image.astype(np.float32)  # 转换为float32

        return cv2.bilateralFilter(image_32f, d=kernel_size, sigmaColor=15, sigmaSpace=15)

def gaussian_lowpass_resize(image, target_size=(64, 64), sigma_factor=0.5):
    """优化后的下采样函数"""
    h, w = image.shape[:2]
    th, tw = target_size

    if len(image.shape) == 3:
        channels = []
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            channels.append(gaussian_lowpass_resize(channel, target_size, sigma_factor))
        return np.stack(channels, axis=-1)

    # 应用窗函数减少边界效应
    # window = np.outer(np.hanning(h), np.hanning(w))
    f = fft2(image )
    f_shift = fftshift(f)

    # 创建高斯滤波器
    crow, ccol = h // 2, w // 2
    sigma = min(h, w) * min(th / h, tw / w) * sigma_factor
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask = np.exp(-(distance ** 2) / (2 * (sigma ** 2)))

    # 应用滤波器
    f_shift_filtered = f_shift * mask
    f_filtered = ifftshift(f_shift_filtered)
    img_filtered = np.abs(ifft2(f_filtered))

    # 后处理减少振铃
    img_filtered = reduce_ringing_artifacts(img_filtered)

    # 归一化和下采样
    img_filtered = (img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min()) * 255
    img_downsampled = cv2.resize(img_filtered.astype(np.uint8), (tw, th), interpolation=cv2.INTER_CUBIC)

    return img_downsampled


def process_images(input_dir, output_dir, target_size=(64, 64)):
    """
    处理目录中的所有图像

    参数:
    input_dir: 输入图像目录
    output_dir: 输出图像目录
    target_size: 目标尺寸 (height, width)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    print(f"开始处理 {len(image_files)} 张图像...")

    # 存储样本用于可视化
    original_samples = []
    downsampled_samples = []
    sample_names = []

    for img_file in tqdm(image_files, desc="处理进度"):
        # 读取图像
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用频域下采样
        img_downsampled = gaussian_lowpass_resize(img_rgb, target_size, sigma_factor=0.35)

        # 保存结果
        output_path = os.path.join(output_dir, f"{img_file}")
        cv2.imwrite(output_path, cv2.cvtColor(img_downsampled, cv2.COLOR_RGB2BGR))

        # 存储样本用于可视化
        if len(original_samples) < 3:
            original_samples.append(img_rgb)
            downsampled_samples.append(img_downsampled)
            sample_names.append(img_file)

    print("处理完成!")

    # 可视化结果
    if original_samples:
        visualize_results(original_samples, downsampled_samples, sample_names, output_dir)


def visualize_results(originals, downsampled, names, output_dir):
    """
    可视化原始图像和下采样结果

    参数:
    originals: 原始图像列表
    downsampled: 下采样图像列表
    names: 图像文件名列表
    output_dir: 输出目录
    """
    num_samples = len(originals)

    plt.figure(figsize=(15, 8))
    plt.suptitle("频域下采样结果 (256x256 → 64x64)", fontsize=16)

    for i in range(num_samples):
        # 原始图像
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(originals[i])
        plt.title(f"原始: {names[i][:15]}...")
        plt.axis('off')

        # 下采样图像
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(downsampled[i])
        plt.title("下采样 (含振铃效应)")
        plt.axis('off')

    plt.tight_layout()

    # 保存可视化结果
    output_vis = os.path.join(output_dir, "downsample_comparison.png")
    plt.savefig(output_vis, dpi=120, bbox_inches='tight')
    print(f"可视化对比图已保存至: {output_vis}")

    # 显示图像
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为你的原始图像目录
    OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_zhenling"  # 输出目录

    # 目标尺寸
    TARGET_SIZE = (64, 64)  # 高度, 宽度

    # 执行处理
    process_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
#
# def downsample_and_add_noise(input_path, output_path, noise_level=25):
#     """
#     将256x256图像下采样至64x64并添加高斯噪声
#
#     参数:
#     input_path: 输入图像目录路径
#     output_path: 输出图像目录路径
#     noise_level: 高斯噪声的标准差 (默认25)
#     """
#     # 创建输出目录
#     os.makedirs(output_path, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_path)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     # 存储处理前后的图像用于可视化
#     pre_process_samples = []
#     post_process_samples = []
#     sample_names = []
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取原始图像
#         img_path = os.path.join(input_path, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             continue
#
#         # 2. 转换色彩空间 (保持RGB格式)
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#         # 3. 双三次下采样至64x64
#         downsampled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
#
#         # 4. 添加高斯噪声
#         # 生成噪声矩阵 (高斯分布)
#         noise = np.random.normal(0, noise_level, downsampled.shape).astype(np.float32)
#
#         # 添加噪声并确保值在0-255范围内
#         noisy_image = downsampled.astype(np.float32) + noise
#         noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#
#         # 5. 保存结果图像
#         output_file = os.path.join(output_path, f"{img_file}")
#         cv2.imwrite(output_file, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
#
#         # 6. 存储样本用于可视化 (最多5个)
#         if len(pre_process_samples) < 5:
#             pre_process_samples.append(downsampled)
#             post_process_samples.append(noisy_image)
#             sample_names.append(img_file)
#
#     print("处理完成!")
#
#     # 7. 可视化处理效果
#     visualize_results(pre_process_samples, post_process_samples, sample_names, noise_level)
#
#
# def visualize_results(original_imgs, processed_imgs, img_names, noise_level):
#     """
#     可视化下采样加噪前后对比
#
#     参数:
#     original_imgs: 下采样后图像列表 (64x64)
#     processed_imgs: 加噪后图像列表 (64x64)
#     img_names: 对应的图像文件名列表
#     noise_level: 使用的噪声水平
#     """
#     num_samples = len(original_imgs)
#
#     plt.figure(figsize=(15, 6))
#
#     for i in range(num_samples):
#         # 原始下采样图像
#         plt.subplot(2, num_samples, i + 1)
#         plt.imshow(original_imgs[i])
#         plt.title(f"Downsampled: {img_names[i][:15]}...")
#         plt.axis('off')
#
#         # 加噪后图像
#         plt.subplot(2, num_samples, num_samples + i + 1)
#         plt.imshow(processed_imgs[i])
#         plt.title(f"Noisy (σ={noise_level})")
#         plt.axis('off')
#
#     plt.suptitle(f"Downsampling (64x64) + Gaussian Noise (σ={noise_level}) Comparison", fontsize=16)
#     plt.tight_layout()
#
#     # 保存可视化结果
#     # output_vis = os.path.join(output_path, "visualization.png")
#     # plt.savefig(output_vis, dpi=120, bbox_inches='tight')
#     # print(f"可视化结果已保存至: {output_vis}")
#
#     # 显示图像
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为你的原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_gussionNoise"  # 输出目录
#
#     # 噪声水平 (标准差) - 25 是常用起点，可根据需要调整
#     NOISE_LEVEL = 1
#
#     # 执行处理
#     downsample_and_add_noise(INPUT_DIR, OUTPUT_DIR, noise_level=NOISE_LEVEL)
# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
#
# def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
#     """
#     向图像添加椒盐噪声 - 部分像素版本
#
#     参数:
#     image: 输入图像 (numpy数组)
#     salt_prob: 盐噪声(白点)的概率
#     pepper_prob: 椒噪声(黑点)的概率
#
#     返回:
#     添加噪声后的图像
#     """
#     noisy = np.copy(image)
#     h, w, c = image.shape
#
#     # 创建噪声强度图
#     salt_intensity = np.random.random((h, w))
#     pepper_intensity = np.random.random((h, w))
#
#     # 应用盐噪声 (部分像素)
#     salt_mask = salt_intensity < salt_prob
#     for i in range(c):
#         noisy[:, :, i][salt_mask] = np.clip(
#             noisy[:, :, i][salt_mask] +
#             salt_intensity[salt_mask] * 100, 0, 255
#         )
#
#     # 应用椒噪声 (部分像素)
#     pepper_mask = pepper_intensity < pepper_prob
#     for i in range(c):
#         noisy[:, :, i][pepper_mask] = np.clip(
#             noisy[:, :, i][pepper_mask] -
#             pepper_intensity[pepper_mask] * 100, 0, 255
#         )
#
#     return noisy
#
#
# def downsample_and_add_salt_pepper(input_path, output_path, salt_prob=0.01, pepper_prob=0.01):
#     """
#     将256x256图像下采样至64x64并添加椒盐噪声
#
#     参数:
#     input_path: 输入图像目录路径
#     output_path: 输出图像目录路径
#     salt_prob: 盐噪声概率 (默认0.01)
#     pepper_prob: 椒噪声概率 (默认0.01)
#     """
#     # 创建输出目录
#     os.makedirs(output_path, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_path)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     # 存储处理前后的图像用于可视化
#     pre_process_samples = []
#     post_process_samples = []
#     sample_names = []
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取原始图像
#         img_path = os.path.join(input_path, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 双三次下采样至64x64
#         downsampled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
#
#         # 3. 添加椒盐噪声
#         noisy_image = add_salt_pepper_noise(downsampled, salt_prob, pepper_prob)
#
#         # 4. 保存结果图像
#         output_file = os.path.join(output_path, f"{img_file}")
#         cv2.imwrite(output_file, noisy_image)
#
#         # 5. 存储样本用于可视化 (最多5个)
#         if len(pre_process_samples) < 5:
#             pre_process_samples.append(downsampled)
#             post_process_samples.append(noisy_image)
#             sample_names.append(img_file)
#
#     print("处理完成!")
#
#     # 6. 可视化处理效果
#     if pre_process_samples:
#         visualize_results(pre_process_samples, post_process_samples, sample_names, salt_prob, pepper_prob, output_path)
#
#
# def visualize_results(original_imgs, processed_imgs, img_names, salt_prob, pepper_prob, output_path):
#     """
#     可视化下采样加噪前后对比
#
#     参数:
#     original_imgs: 下采样后图像列表 (64x64)
#     processed_imgs: 加噪后图像列表 (64x64)
#     img_names: 对应的图像文件名列表
#     salt_prob: 盐噪声概率
#     pepper_prob: 椒噪声概率
#     output_path: 输出目录路径
#     """
#     num_samples = len(original_imgs)
#
#     plt.figure(figsize=(15, 6))
#     plt.suptitle(f"Downsampling (64x64) + Salt & Pepper Noise (Salt: {salt_prob}, Pepper: {pepper_prob})", fontsize=16)
#
#     for i in range(num_samples):
#         # 原始下采样图像
#         plt.subplot(2, num_samples, i + 1)
#         plt.imshow(cv2.cvtColor(original_imgs[i], cv2.COLOR_BGR2RGB))
#         plt.title(f"Downsampled: {img_names[i][:15]}...")
#         plt.axis('off')
#
#         # 加噪后图像
#         plt.subplot(2, num_samples, num_samples + i + 1)
#         plt.imshow(cv2.cvtColor(processed_imgs[i], cv2.COLOR_BGR2RGB))
#         plt.title(f"Salt & Pepper Noise")
#         plt.axis('off')
#
#     plt.tight_layout()
#
#     # 保存可视化结果
#     output_vis = os.path.join(output_path, "salt_pepper_comparison.png")
#     plt.savefig(output_vis, dpi=120, bbox_inches='tight')
#     print(f"可视化对比图已保存至: {output_vis}")
#
#     # 显示图像
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为你的原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_jiaoyanNoise"  # 输出目录
#
#     # 噪声参数 (可根据需要调整)
#     SALT_PROB = 0.1  # 盐噪声概率 (白点)
#     PEPPER_PROB = 0.1  # 椒噪声概率 (黑点)
#
#     # 执行处理
#     downsample_and_add_salt_pepper(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         salt_prob=SALT_PROB,
#         pepper_prob=PEPPER_PROB
#     )
# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
#
# def add_rayleigh_noise(image, scale=30):
#     """
#     向图像添加瑞利噪声
#
#     参数:
#     image: 输入图像 (numpy数组)
#     scale: 瑞利分布的尺度参数 (控制噪声强度)
#
#     返回:
#     添加噪声后的图像
#     """
#     # 生成瑞利噪声
#     noise = np.random.rayleigh(scale, size=image.shape)
#
#     # 将噪声添加到图像
#     noisy_image = image.astype(np.float32) + noise
#
#     # 确保像素值在0-255范围内
#     noisy_image = np.clip(noisy_image, 0, 255)
#
#     return noisy_image.astype(np.uint8)
#
#
# def downsample_and_add_rayleigh_noise(input_path, output_path, scale=30):
#     """
#     将256x256图像下采样至64x64并添加瑞利噪声
#
#     参数:
#     input_path: 输入图像目录路径
#     output_path: 输出图像目录路径
#     scale: 瑞利噪声的尺度参数 (默认30)
#     """
#     # 创建输出目录
#     os.makedirs(output_path, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_path)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     # 存储处理前后的图像用于可视化
#     pre_process_samples = []
#     post_process_samples = []
#     sample_names = []
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取原始图像
#         img_path = os.path.join(input_path, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 双三次下采样至64x64
#         downsampled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
#
#         # 3. 添加瑞利噪声
#         noisy_image = add_rayleigh_noise(downsampled, scale)
#
#         # 4. 保存结果图像
#         output_file = os.path.join(output_path, f"{img_file}")
#         cv2.imwrite(output_file, noisy_image)
#
#         # 5. 存储样本用于可视化 (最多5个)
#         if len(pre_process_samples) < 5:
#             pre_process_samples.append(downsampled)
#             post_process_samples.append(noisy_image)
#             sample_names.append(img_file)
#
#     print("处理完成!")
#
#     # 6. 可视化处理效果
#     if pre_process_samples:
#         visualize_results(pre_process_samples, post_process_samples, sample_names, scale, output_path)
#
#
# def visualize_results(original_imgs, processed_imgs, img_names, scale, output_path):
#     """
#     可视化下采样加噪前后对比
#
#     参数:
#     original_imgs: 下采样后图像列表 (64x64)
#     processed_imgs: 加噪后图像列表 (64x64)
#     img_names: 对应的图像文件名列表
#     scale: 瑞利噪声尺度参数
#     output_path: 输出目录路径
#     """
#     num_samples = len(original_imgs)
#
#     plt.figure(figsize=(15, 6))
#     plt.suptitle(f"Downsampling (64x64) + Rayleigh Noise (Scale={scale})", fontsize=16)
#
#     for i in range(num_samples):
#         # 原始下采样图像
#         plt.subplot(2, num_samples, i + 1)
#         plt.imshow(cv2.cvtColor(original_imgs[i], cv2.COLOR_BGR2RGB))
#         plt.title(f"Downsampled: {img_names[i][:15]}...")
#         plt.axis('off')
#
#         # 加噪后图像
#         plt.subplot(2, num_samples, num_samples + i + 1)
#         plt.imshow(cv2.cvtColor(processed_imgs[i], cv2.COLOR_BGR2RGB))
#         plt.title(f"Rayleigh Noise")
#         plt.axis('off')
#
#     plt.tight_layout()
#
#     # 保存可视化结果
#     output_vis = os.path.join(output_path, "rayleigh_noise_comparison.png")
#     plt.savefig(output_vis, dpi=120, bbox_inches='tight')
#     print(f"可视化对比图已保存至: {output_vis}")
#
#     # 显示图像
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为你的原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_ruiliNoise"  # 输出目录
#
#     # 噪声参数 (可根据需要调整)
#     RAYLEIGH_SCALE = 3  # 瑞利噪声尺度参数
#
#     # 执行处理
#     downsample_and_add_rayleigh_noise(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         scale=RAYLEIGH_SCALE
#     )
# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
#
# def add_blocking_artifacts(image, block_size=8, compression_strength=30):
#     """
#     添加分块效应到图像
#
#     参数:
#     image: 输入图像 (numpy数组)
#     block_size: 分块大小 (默认8x8)
#     compression_strength: 压缩强度 (0-100)
#
#     返回:
#     添加分块效应后的图像
#     """
#     # 如果是彩色图像，分别处理每个通道
#     if len(image.shape) == 3:
#         channels = []
#         for c in range(image.shape[2]):
#             channel = image[:, :, c]
#             channels.append(add_blocking_artifacts(channel, block_size, compression_strength))
#         return np.stack(channels, axis=-1)
#
#     # 获取图像尺寸
#     h, w = image.shape
#
#     # 创建分块网格
#     grid = np.zeros((h, w))
#
#     # 计算块边界强度
#     boundary_strength = compression_strength / 100.0 * 50
#
#     # 添加水平边界线
#     for i in range(0, h, block_size):
#         if i > 0:
#             grid[i - 1:i + 1, :] = boundary_strength
#
#     # 添加垂直边界线
#     for j in range(0, w, block_size):
#         if j > 0:
#             grid[:, j - 1:j + 1] = boundary_strength
#
#     # 添加随机块内噪声
#     noise = np.random.normal(0, compression_strength / 5, (h, w))
#
#     # 应用分块效应
#     noisy_image = image.astype(np.float32) + grid + noise
#
#     # 裁剪到有效范围
#     noisy_image = np.clip(noisy_image, 0, 255)
#
#     return noisy_image.astype(np.uint8)
#
#
# def downsample_with_blocking(input_dir, output_dir, target_size=(64, 64), block_size=8, compression_strength=30):
#     """
#     处理目录中的所有图像，添加分块效应并下采样
#
#     参数:
#     input_dir: 输入图像目录
#     output_dir: 输出图像目录
#     target_size: 目标尺寸 (height, width)
#     block_size: 分块大小
#     compression_strength: 压缩强度 (0-100)
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     # 存储样本用于可视化
#     original_samples = []
#     downsampled_samples = []
#     block_samples = []
#     sample_names = []
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 读取图像
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 转换为RGB格式
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # 1. 添加分块效应
#         img_blocked = add_blocking_artifacts(img_rgb, block_size, compression_strength)
#
#         # 2. 下采样到目标尺寸
#         img_downsampled = cv2.resize(img_blocked, target_size[::-1], interpolation=cv2.INTER_CUBIC)
#
#         # 保存结果
#         output_path = os.path.join(output_dir, f"{img_file}")
#         cv2.imwrite(output_path, cv2.cvtColor(img_downsampled, cv2.COLOR_RGB2BGR))
#
#         # 存储样本用于可视化
#         if len(original_samples) < 3:
#             original_samples.append(img_rgb)
#             block_samples.append(img_blocked)
#             downsampled_samples.append(img_downsampled)
#             sample_names.append(img_file)
#
#     print("处理完成!")
#
#     # 可视化结果
#     if original_samples:
#         visualize_results(original_samples, block_samples, downsampled_samples, sample_names, output_dir)
#
#
# def visualize_results(originals, blocked, downsampled, names, output_dir):
#     """
#     可视化处理各阶段结果
#
#     参数:
#     originals: 原始图像列表
#     blocked: 添加分块效应后的图像列表
#     downsampled: 下采样图像列表
#     names: 图像文件名列表
#     output_dir: 输出目录
#     """
#     num_samples = len(originals)
#
#     plt.figure(figsize=(15, 10))
#     plt.suptitle(f"图像处理流程: 原始 → 分块效应 → 下采样", fontsize=16)
#
#     for i in range(num_samples):
#         # 原始图像 (256x256)
#         plt.subplot(3, num_samples, i + 1)
#         plt.imshow(originals[i])
#         plt.title(f"原始: {names[i][:15]}...")
#         plt.axis('off')
#
#         # 分块效应图像 (256x256)
#         plt.subplot(3, num_samples, num_samples + i + 1)
#         plt.imshow(blocked[i])
#         plt.title("分块效应")
#         plt.axis('off')
#
#         # 下采样图像 (64x64)
#         plt.subplot(3, num_samples, 2 * num_samples + i + 1)
#         plt.imshow(downsampled[i])
#         plt.title("下采样 (64x64)")
#         plt.axis('off')
#
#     plt.tight_layout()
#
#     # 保存可视化结果
#     output_vis = os.path.join(output_dir, "blocking_comparison.png")
#     plt.savefig(output_vis, dpi=120, bbox_inches='tight')
#     print(f"可视化对比图已保存至: {output_vis}")
#
#     # 显示图像
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_blocking"  # 输出目录
#
#     # 目标尺寸
#     TARGET_SIZE = (64, 64)  # 高度, 宽度
#
#     # 分块参数
#     BLOCK_SIZE = 128 # 分块大小 (通常8x8)
#     COMPRESSION_STRENGTH = 100  # 压缩强度 (0-100)
#
#     # 执行处理
#     downsample_with_blocking(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         target_size=TARGET_SIZE,
#         block_size=BLOCK_SIZE,
#         compression_strength=COMPRESSION_STRENGTH
#     )
# # -*- coding: utf-8 -*-
# import numpy as np
# import cv2
# import torch
#
# import random
# from scipy import ndimage
# import scipy
# import scipy.stats as ss
# from scipy.interpolate import interp2d
# from scipy.linalg import orth
# import os
# import math
# import random
# import numpy as np
# import torch
# import cv2
# from torchvision.utils import make_grid
# from datetime import datetime
# # import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from tqdm import tqdm
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# '''
# # --------------------------------------------
# # Kai Zhang (github: https://github.com/cszn)
# # 03/Mar/2019
# # --------------------------------------------
# # https://github.com/twhui/SRGAN-pyTorch
# # https://github.com/xinntao/BasicSR
# # --------------------------------------------
# '''
#
# IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
#
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
#
# def get_timestamp():
#     return datetime.now().strftime('%y%m%d-%H%M%S')
#
#
# def imshow(x, title=None, cbar=False, figsize=None):
#     plt.figure(figsize=figsize)
#     plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
#     if title:
#         plt.title(title)
#     if cbar:
#         plt.colorbar()
#     plt.show()
#
#
# def surf(Z, cmap='rainbow', figsize=None):
#     plt.figure(figsize=figsize)
#     ax3 = plt.axes(projection='3d')
#
#     w, h = Z.shape[:2]
#     xx = np.arange(0, w, 1)
#     yy = np.arange(0, h, 1)
#     X, Y = np.meshgrid(xx, yy)
#     ax3.plot_surface(X, Y, Z, cmap=cmap)
#     # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
#     plt.show()
#
#
# '''
# # --------------------------------------------
# # get image pathes
# # --------------------------------------------
# '''
#
#
# def get_image_paths(dataroot):
#     paths = None  # return None if dataroot is None
#     if dataroot is not None:
#         paths = sorted(_get_paths_from_images(dataroot))
#     return paths
#
#
# def _get_paths_from_images(path):
#     assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
#     images = []
#     for dirpath, _, fnames in sorted(os.walk(path)):
#         for fname in sorted(fnames):
#             if is_image_file(fname):
#                 img_path = os.path.join(dirpath, fname)
#                 images.append(img_path)
#     assert images, '{:s} has no valid image file'.format(path)
#     return images
#
#
# '''
# # --------------------------------------------
# # split large images into small images
# # --------------------------------------------
# '''
#
#
# def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
#     w, h = img.shape[:2]
#     patches = []
#     if w > p_max and h > p_max:
#         w1 = list(np.arange(0, w - p_size, p_size - p_overlap, dtype=np.int))
#         h1 = list(np.arange(0, h - p_size, p_size - p_overlap, dtype=np.int))
#         w1.append(w - p_size)
#         h1.append(h - p_size)
#         #        print(w1)
#         #        print(h1)
#         for i in w1:
#             for j in h1:
#                 patches.append(img[i:i + p_size, j:j + p_size, :])
#     else:
#         patches.append(img)
#
#     return patches
#
#
# def imssave(imgs, img_path):
#     """
#     imgs: list, N images of size WxHxC
#     """
#     img_name, ext = os.path.splitext(os.path.basename(img_path))
#
#     for i, img in enumerate(imgs):
#         if img.ndim == 3:
#             img = img[:, :, [2, 1, 0]]
#         new_path = os.path.join(os.path.dirname(img_path), img_name + str('_s{:04d}'.format(i)) + '.png')
#         cv2.imwrite(new_path, img)
#
#
# def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=800, p_overlap=96, p_max=1000):
#     """
#     split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size),
#     and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
#     will be splitted.
#
#     Args:
#         original_dataroot:
#         taget_dataroot:
#         p_size: size of small images
#         p_overlap: patch size in training is a good choice
#         p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
#     """
#     paths = get_image_paths(original_dataroot)
#     for img_path in paths:
#         # img_name, ext = os.path.splitext(os.path.basename(img_path))
#         img = imread_uint(img_path, n_channels=n_channels)
#         patches = patches_from_image(img, p_size, p_overlap, p_max)
#         imssave(patches, os.path.join(taget_dataroot, os.path.basename(img_path)))
#         # if original_dataroot == taget_dataroot:
#         # del img_path
#
#
# '''
# # --------------------------------------------
# # makedir
# # --------------------------------------------
# '''
#
#
# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#
# def mkdirs(paths):
#     if isinstance(paths, str):
#         mkdir(paths)
#     else:
#         for path in paths:
#             mkdir(path)
#
#
# def mkdir_and_rename(path):
#     if os.path.exists(path):
#         new_name = path + '_archived_' + get_timestamp()
#         print('Path already exists. Rename it to [{:s}]'.format(new_name))
#         os.rename(path, new_name)
#     os.makedirs(path)
#
#
# '''
# # --------------------------------------------
# # read image from path
# # opencv is fast, but read BGR numpy image
# # --------------------------------------------
# '''
#
#
# # --------------------------------------------
# # get uint8 image of size HxWxn_channles (RGB)
# # --------------------------------------------
# def imread_uint(path, n_channels=3):
#     #  input: path
#     # output: HxWx3(RGB or GGG), or HxWx1 (G)
#     if n_channels == 1:
#         img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
#         img = np.expand_dims(img, axis=2)  # HxWx1
#     elif n_channels == 3:
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
#         # if img.ndim == 2:
#         #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
#         # else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
#     return img
#
#
# # --------------------------------------------
# # matlab's imwrite
# # --------------------------------------------
# def imsave(img, img_path):
#     img = np.squeeze(img)
#     if img.ndim == 3:
#         img = img[:, :, [2, 1, 0]]
#     cv2.imwrite(img_path, img)
#
#
# def imwrite(img, img_path):
#     img = np.squeeze(img)
#     if img.ndim == 3:
#         img = img[:, :, [2, 1, 0]]
#     cv2.imwrite(img_path, img)
#
#
# # --------------------------------------------
# # get single image of size HxWxn_channles (BGR)
# # --------------------------------------------
# def read_img(path):
#     # read image by cv2
#     # return: Numpy float32, HWC, BGR, [0,1]
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
#     img = img.astype(np.float32) / 255.
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     # some images have 4 channels
#     if img.shape[2] > 3:
#         img = img[:, :, :3]
#     return img
#
#
# '''
# # --------------------------------------------
# # image format conversion
# # --------------------------------------------
# # numpy(single) <--->  numpy(unit)
# # numpy(single) <--->  tensor
# # numpy(unit)   <--->  tensor
# # --------------------------------------------
# '''
#
#
# # --------------------------------------------
# # numpy(single) [0, 1] <--->  numpy(unit)
# # --------------------------------------------
#
#
# def uint2single(img):
#     return np.float32(img / 255.)
#
#
# def single2uint(img):
#     return np.uint8((img.clip(0, 1) * 255.).round())
#
#
# def uint162single(img):
#     return np.float32(img / 65535.)
#
#
# def single2uint16(img):
#     return np.uint16((img.clip(0, 1) * 65535.).round())
#
#
# # --------------------------------------------
# # numpy(unit) (HxWxC or HxW) <--->  tensor
# # --------------------------------------------
#
#
# # convert uint to 4-dimensional torch tensor
# def uint2tensor4(img):
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
#
#
# # convert uint to 3-dimensional torch tensor
# def uint2tensor3(img):
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)
#
#
# # convert 2/3/4-dimensional torch tensor to uint
# def tensor2uint(img):
#     img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
#     if img.ndim == 3:
#         img = np.transpose(img, (1, 2, 0))
#     return np.uint8((img * 255.0).round())
#
#
# # --------------------------------------------
# # numpy(single) (HxWxC) <--->  tensor
# # --------------------------------------------
#
#
# # convert single (HxWxC) to 3-dimensional torch tensor
# def single2tensor3(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
#
#
# # convert single (HxWxC) to 4-dimensional torch tensor
# def single2tensor4(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
#
#
# # convert torch tensor to single
# def tensor2single(img):
#     img = img.data.squeeze().float().cpu().numpy()
#     if img.ndim == 3:
#         img = np.transpose(img, (1, 2, 0))
#
#     return img
#
#
# # convert torch tensor to single
# def tensor2single3(img):
#     img = img.data.squeeze().float().cpu().numpy()
#     if img.ndim == 3:
#         img = np.transpose(img, (1, 2, 0))
#     elif img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     return img
#
#
# def single2tensor5(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)
#
#
# def single32tensor5(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)
#
#
# def single42tensor4(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()
#
#
# # from skimage.io import imread, imsave
# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array of BGR channel order
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # squeeze first, then clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type)
#
#
# '''
# # --------------------------------------------
# # Augmentation, flipe and/or rotate
# # --------------------------------------------
# # The following two are enough.
# # (1) augmet_img: numpy image of WxHxC or WxH
# # (2) augment_img_tensor4: tensor image 1xCxWxH
# # --------------------------------------------
# '''
#
#
# def augment_img(img, mode=0):
#     '''Kai Zhang (github: https://github.com/cszn)
#     '''
#     if mode == 0:
#         return img
#     elif mode == 1:
#         return np.flipud(np.rot90(img))
#     elif mode == 2:
#         return np.flipud(img)
#     elif mode == 3:
#         return np.rot90(img, k=3)
#     elif mode == 4:
#         return np.flipud(np.rot90(img, k=2))
#     elif mode == 5:
#         return np.rot90(img)
#     elif mode == 6:
#         return np.rot90(img, k=2)
#     elif mode == 7:
#         return np.flipud(np.rot90(img, k=3))
#
#
# def augment_img_tensor4(img, mode=0):
#     '''Kai Zhang (github: https://github.com/cszn)
#     '''
#     if mode == 0:
#         return img
#     elif mode == 1:
#         return img.rot90(1, [2, 3]).flip([2])
#     elif mode == 2:
#         return img.flip([2])
#     elif mode == 3:
#         return img.rot90(3, [2, 3])
#     elif mode == 4:
#         return img.rot90(2, [2, 3]).flip([2])
#     elif mode == 5:
#         return img.rot90(1, [2, 3])
#     elif mode == 6:
#         return img.rot90(2, [2, 3])
#     elif mode == 7:
#         return img.rot90(3, [2, 3]).flip([2])
#
#
# def augment_img_tensor(img, mode=0):
#     '''Kai Zhang (github: https://github.com/cszn)
#     '''
#     img_size = img.size()
#     img_np = img.data.cpu().numpy()
#     if len(img_size) == 3:
#         img_np = np.transpose(img_np, (1, 2, 0))
#     elif len(img_size) == 4:
#         img_np = np.transpose(img_np, (2, 3, 1, 0))
#     img_np = augment_img(img_np, mode=mode)
#     img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
#     if len(img_size) == 3:
#         img_tensor = img_tensor.permute(2, 0, 1)
#     elif len(img_size) == 4:
#         img_tensor = img_tensor.permute(3, 2, 0, 1)
#
#     return img_tensor.type_as(img)
#
#
# def augment_img_np3(img, mode=0):
#     if mode == 0:
#         return img
#     elif mode == 1:
#         return img.transpose(1, 0, 2)
#     elif mode == 2:
#         return img[::-1, :, :]
#     elif mode == 3:
#         img = img[::-1, :, :]
#         img = img.transpose(1, 0, 2)
#         return img
#     elif mode == 4:
#         return img[:, ::-1, :]
#     elif mode == 5:
#         img = img[:, ::-1, :]
#         img = img.transpose(1, 0, 2)
#         return img
#     elif mode == 6:
#         img = img[:, ::-1, :]
#         img = img[::-1, :, :]
#         return img
#     elif mode == 7:
#         img = img[:, ::-1, :]
#         img = img[::-1, :, :]
#         img = img.transpose(1, 0, 2)
#         return img
#
#
# def augment_imgs(img_list, hflip=True, rot=True):
#     # horizontal flip OR rotate
#     hflip = hflip and random.random() < 0.5
#     vflip = rot and random.random() < 0.5
#     rot90 = rot and random.random() < 0.5
#
#     def _augment(img):
#         if hflip:
#             img = img[:, ::-1, :]
#         if vflip:
#             img = img[::-1, :, :]
#         if rot90:
#             img = img.transpose(1, 0, 2)
#         return img
#
#     return [_augment(img) for img in img_list]
#
#
# '''
# # --------------------------------------------
# # modcrop and shave
# # --------------------------------------------
# '''
#
#
# def modcrop(img_in, scale):
#     # img_in: Numpy, HWC or HW
#     img = np.copy(img_in)
#     if img.ndim == 2:
#         H, W = img.shape
#         H_r, W_r = H % scale, W % scale
#         img = img[:H - H_r, :W - W_r]
#     elif img.ndim == 3:
#         H, W, C = img.shape
#         H_r, W_r = H % scale, W % scale
#         img = img[:H - H_r, :W - W_r, :]
#     else:
#         raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
#     return img
#
#
# def shave(img_in, border=0):
#     # img_in: Numpy, HWC or HW
#     img = np.copy(img_in)
#     h, w = img.shape[:2]
#     img = img[border:h - border, border:w - border]
#     return img
#
#
# '''
# # --------------------------------------------
# # image processing process on numpy image
# # channel_convert(in_c, tar_type, img_list):
# # rgb2ycbcr(img, only_y=True):
# # bgr2ycbcr(img, only_y=True):
# # ycbcr2rgb(img):
# # --------------------------------------------
# '''
#
#
# def rgb2ycbcr(img, only_y=True):
#     '''same as matlab rgb2ycbcr
#     only_y: only return Y channel
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     if only_y:
#         rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
#     else:
#         rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
#                               [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)
#
#
# def ycbcr2rgb(img):
#     '''same as matlab ycbcr2rgb
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
#                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)
#
#
# def bgr2ycbcr(img, only_y=True):
#     '''bgr version of rgb2ycbcr
#     only_y: only return Y channel
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     if only_y:
#         rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
#     else:
#         rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
#                               [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)
#
#
# def channel_convert(in_c, tar_type, img_list):
#     # conversion among BGR, gray and y
#     if in_c == 3 and tar_type == 'gray':  # BGR to gray
#         gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
#         return [np.expand_dims(img, axis=2) for img in gray_list]
#     elif in_c == 3 and tar_type == 'y':  # BGR to y
#         y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
#         return [np.expand_dims(img, axis=2) for img in y_list]
#     elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
#         return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
#     else:
#         return img_list
#
#
# '''
# # --------------------------------------------
# # metric, PSNR and SSIM
# # --------------------------------------------
# '''
#
#
# # --------------------------------------------
# # PSNR
# # --------------------------------------------
# def calculate_psnr(img1, img2, border=0):
#     # img1 and img2 have range [0, 255]
#     # img1 = img1.squeeze()
#     # img2 = img2.squeeze()
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h - border, border:w - border]
#     img2 = img2[border:h - border, border:w - border]
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))
#
#
# # --------------------------------------------
# # SSIM
# # --------------------------------------------
# def calculate_ssim(img1, img2, border=0):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     # img1 = img1.squeeze()
#     # img2 = img2.squeeze()
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h - border, border:w - border]
#     img2 = img2[border:h - border, border:w - border]
#
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
#
#
# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
#
# '''
# # --------------------------------------------
# # matlab's bicubic imresize (numpy and torch) [0, 1]
# # --------------------------------------------
# '''
#
#
# # matlab 'imresize' function, now only support 'bicubic'
# def cubic(x):
#     absx = torch.abs(x)
#     absx2 = absx ** 2
#     absx3 = absx ** 3
#     return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
#         (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))
#
#
# def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
#     if (scale < 1) and (antialiasing):
#         # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
#         kernel_width = kernel_width / scale
#
#     # Output-space coordinates
#     x = torch.linspace(1, out_length, out_length)
#
#     # Input-space coordinates. Calculate the inverse mapping such that 0.5
#     # in output space maps to 0.5 in input space, and 0.5+scale in output
#     # space maps to 1.5 in input space.
#     u = x / scale + 0.5 * (1 - 1 / scale)
#
#     # What is the left-most pixel that can be involved in the computation?
#     left = torch.floor(u - kernel_width / 2)
#
#     # What is the maximum number of pixels that can be involved in the
#     # computation?  Note: it's OK to use an extra pixel here; if the
#     # corresponding weights are all zero, it will be eliminated at the end
#     # of this function.
#     P = math.ceil(kernel_width) + 2
#
#     # The indices of the input pixels involved in computing the k-th output
#     # pixel are in row k of the indices matrix.
#     indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
#         1, P).expand(out_length, P)
#
#     # The weights used to compute the k-th output pixel are in row k of the
#     # weights matrix.
#     distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
#     # apply cubic kernel
#     if (scale < 1) and (antialiasing):
#         weights = scale * cubic(distance_to_center * scale)
#     else:
#         weights = cubic(distance_to_center)
#     # Normalize the weights matrix so that each row sums to 1.
#     weights_sum = torch.sum(weights, 1).view(out_length, 1)
#     weights = weights / weights_sum.expand(out_length, P)
#
#     # If a column in weights is all zero, get rid of it. only consider the first and last column.
#     weights_zero_tmp = torch.sum((weights == 0), 0)
#     if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
#         indices = indices.narrow(1, 1, P - 2)
#         weights = weights.narrow(1, 1, P - 2)
#     if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
#         indices = indices.narrow(1, 0, P - 2)
#         weights = weights.narrow(1, 0, P - 2)
#     weights = weights.contiguous()
#     indices = indices.contiguous()
#     sym_len_s = -indices.min() + 1
#     sym_len_e = indices.max() - in_length
#     indices = indices + sym_len_s - 1
#     return weights, indices, int(sym_len_s), int(sym_len_e)
#
#
# # --------------------------------------------
# # imresize for tensor image [0, 1]
# # --------------------------------------------
# def imresize(img, scale, antialiasing=True):
#     # Now the scale should be the same for H and W
#     # input: img: pytorch tensor, CHW or HW [0,1]
#     # output: CHW or HW [0,1] w/o round
#     need_squeeze = True if img.dim() == 2 else False
#     if need_squeeze:
#         img.unsqueeze_(0)
#     in_C, in_H, in_W = img.size()
#     out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
#     kernel_width = 4
#     kernel = 'cubic'
#
#     # Return the desired dimension order for performing the resize.  The
#     # strategy is to perform the resize first along the dimension with the
#     # smallest scale factor.
#     # Now we do not support this.
#
#     # get weights and indices
#     weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
#         in_H, out_H, scale, kernel, kernel_width, antialiasing)
#     weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
#         in_W, out_W, scale, kernel, kernel_width, antialiasing)
#     # process H dimension
#     # symmetric copying
#     img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
#     img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)
#
#     sym_patch = img[:, :sym_len_Hs, :]
#     inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(1, inv_idx)
#     img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)
#
#     sym_patch = img[:, -sym_len_He:, :]
#     inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(1, inv_idx)
#     img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)
#
#     out_1 = torch.FloatTensor(in_C, out_H, in_W)
#     kernel_width = weights_H.size(1)
#     for i in range(out_H):
#         idx = int(indices_H[i][0])
#         for j in range(out_C):
#             out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
#
#     # process W dimension
#     # symmetric copying
#     out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
#     out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)
#
#     sym_patch = out_1[:, :, :sym_len_Ws]
#     inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(2, inv_idx)
#     out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)
#
#     sym_patch = out_1[:, :, -sym_len_We:]
#     inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(2, inv_idx)
#     out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)
#
#     out_2 = torch.FloatTensor(in_C, out_H, out_W)
#     kernel_width = weights_W.size(1)
#     for i in range(out_W):
#         idx = int(indices_W[i][0])
#         for j in range(out_C):
#             out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_W[i])
#     if need_squeeze:
#         out_2.squeeze_()
#     return out_2
#
#
# # --------------------------------------------
# # imresize for numpy image [0, 1]
# # --------------------------------------------
# def imresize_np(img, scale, antialiasing=True):
#     # Now the scale should be the same for H and W
#     # input: img: Numpy, HWC or HW [0,1]
#     # output: HWC or HW [0,1] w/o round
#     img = torch.from_numpy(img)
#     need_squeeze = True if img.dim() == 2 else False
#     if need_squeeze:
#         img.unsqueeze_(2)
#
#     in_H, in_W, in_C = img.size()
#     out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
#     kernel_width = 4
#     kernel = 'cubic'
#
#     # Return the desired dimension order for performing the resize.  The
#     # strategy is to perform the resize first along the dimension with the
#     # smallest scale factor.
#     # Now we do not support this.
#
#     # get weights and indices
#     weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
#         in_H, out_H, scale, kernel, kernel_width, antialiasing)
#     weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
#         in_W, out_W, scale, kernel, kernel_width, antialiasing)
#     # process H dimension
#     # symmetric copying
#     img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
#     img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)
#
#     sym_patch = img[:sym_len_Hs, :, :]
#     inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(0, inv_idx)
#     img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)
#
#     sym_patch = img[-sym_len_He:, :, :]
#     inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(0, inv_idx)
#     img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)
#
#     out_1 = torch.FloatTensor(out_H, in_W, in_C)
#     kernel_width = weights_H.size(1)
#     for i in range(out_H):
#         idx = int(indices_H[i][0])
#         for j in range(out_C):
#             out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])
#
#     # process W dimension
#     # symmetric copying
#     out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
#     out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)
#
#     sym_patch = out_1[:, :sym_len_Ws, :]
#     inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(1, inv_idx)
#     out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)
#
#     sym_patch = out_1[:, -sym_len_We:, :]
#     inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#     sym_patch_inv = sym_patch.index_select(1, inv_idx)
#     out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)
#
#     out_2 = torch.FloatTensor(out_H, out_W, in_C)
#     kernel_width = weights_W.size(1)
#     for i in range(out_W):
#         idx = int(indices_W[i][0])
#         for j in range(out_C):
#             out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
#     if need_squeeze:
#         out_2.squeeze_()
#
#     return out_2.numpy()
#
#
# if __name__ == '__main__':
#     print('---')
# #    img = imread_uint('test.bmp', 3)
# #    img = uint2single(img)
# #    img_bicubic = imresize_np(img, 1/4)
#
#
# """
# # --------------------------------------------
# # Super-Resolution
# # --------------------------------------------
# #
# # Kai Zhang (cskaizhang@gmail.com)
# # https://github.com/cszn
# # From 2019/03--2021/08
# # --------------------------------------------
# """
#
#
# def modcrop_np(img, sf):
#     '''
#     Args:
#         img: numpy image, WxH or WxHxC
#         sf: scale factor
#
#     Return:
#         cropped image
#     '''
#     w, h = img.shape[:2]
#     im = np.copy(img)
#     return im[:w - w % sf, :h - h % sf, ...]
#
#
# """
# # --------------------------------------------
# # anisotropic Gaussian kernels
# # --------------------------------------------
# """
#
#
# def analytic_kernel(k):
#     """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
#     k_size = k.shape[0]
#     # Calculate the big kernels size
#     big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
#     # Loop over the small kernel to fill the big one
#     for r in range(k_size):
#         for c in range(k_size):
#             big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
#     # Crop the edges of the big kernel to ignore very small values and increase run time of SR
#     crop = k_size // 2
#     cropped_big_k = big_k[crop:-crop, crop:-crop]
#     # Normalize to 1
#     return cropped_big_k / cropped_big_k.sum()
#
#
# def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
#     """ generate an anisotropic Gaussian kernel
#     Args:
#         ksize : e.g., 15, kernel size
#         theta : [0,  pi], rotation angle range
#         l1    : [0.1,50], scaling of eigenvalues
#         l2    : [0.1,l1], scaling of eigenvalues
#         If l1 = l2, will get an isotropic Gaussian kernel.
#
#     Returns:
#         k     : kernel
#     """
#
#     v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
#     V = np.array([[v[0], v[1]], [v[1], -v[0]]])
#     D = np.array([[l1, 0], [0, l2]])
#     Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
#     k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)
#
#     return k
#
#
# def gm_blur_kernel(mean, cov, size=15):
#     center = size / 2.0 + 0.5
#     k = np.zeros([size, size])
#     for y in range(size):
#         for x in range(size):
#             cy = y - center + 1
#             cx = x - center + 1
#             k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
#
#     k = k / np.sum(k)
#     return k
#
#
# def shift_pixel(x, sf, upper_left=True):
#     """shift pixel for super-resolution with different scale factors
#     Args:
#         x: WxHxC or WxH
#         sf: scale factor
#         upper_left: shift direction
#     """
#     h, w = x.shape[:2]
#     shift = (sf - 1) * 0.5
#     xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
#     if upper_left:
#         x1 = xv + shift
#         y1 = yv + shift
#     else:
#         x1 = xv - shift
#         y1 = yv - shift
#
#     x1 = np.clip(x1, 0, w - 1)
#     y1 = np.clip(y1, 0, h - 1)
#
#     if x.ndim == 2:
#         x = interp2d(xv, yv, x)(x1, y1)
#     if x.ndim == 3:
#         for i in range(x.shape[-1]):
#             x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
#
#     return x
#
#
# def blur(x, k):
#     '''
#     x: image, NxcxHxW
#     k: kernel, Nx1xhxw
#     '''
#     n, c = x.shape[:2]
#     p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
#     x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
#     k = k.repeat(1, c, 1, 1)
#     k = k.view(-1, 1, k.shape[2], k.shape[3])
#     x = x.view(1, -1, x.shape[2], x.shape[3])
#     x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n * c)
#     x = x.view(n, c, x.shape[2], x.shape[3])
#
#     return x
#
#
# def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
#     """"
#     # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
#     # Kai Zhang
#     # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
#     # max_var = 2.5 * sf
#     """
#     # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
#     lambda_1 = min_var + np.random.rand() * (max_var - min_var)
#     lambda_2 = min_var + np.random.rand() * (max_var - min_var)
#     theta = np.random.rand() * np.pi  # random theta
#     noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
#
#     # Set COV matrix using Lambdas and Theta
#     LAMBDA = np.diag([lambda_1, lambda_2])
#     Q = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     SIGMA = Q @ LAMBDA @ Q.T
#     INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
#
#     # Set expectation position (shifting kernel for aligned image)
#     MU = k_size // 2 - 0.5 * (scale_factor - 1)  # - 0.5 * (scale_factor - k_size % 2)
#     MU = MU[None, None, :, None]
#
#     # Create meshgrid for Gaussian
#     [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
#     Z = np.stack([X, Y], 2)[:, :, :, None]
#
#     # Calcualte Gaussian for every pixel of the kernel
#     ZZ = Z - MU
#     ZZ_t = ZZ.transpose(0, 1, 3, 2)
#     raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
#
#     # shift the kernel so it will be centered
#     # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
#
#     # Normalize the kernel and return
#     # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
#     kernel = raw_kernel / np.sum(raw_kernel)
#     return kernel
#
#
# def fspecial_gaussian(hsize, sigma):
#     hsize = [hsize, hsize]
#     siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
#     std = sigma
#     [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
#     arg = -(x * x + y * y) / (2 * std * std)
#     h = np.exp(arg)
#     # h[h < scipy.finfo(float).eps * h.max()] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h = h / sumh
#     return h
#
#
# def fspecial_laplacian(alpha):
#     alpha = max([0, min([alpha, 1])])
#     h1 = alpha / (alpha + 1)
#     h2 = (1 - alpha) / (alpha + 1)
#     h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
#     h = np.array(h)
#     return h
#
#
# def fspecial(filter_type, *args, **kwargs):
#     '''
#     python code from:
#     https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
#     '''
#     if filter_type == 'gaussian':
#         return fspecial_gaussian(*args, **kwargs)
#     if filter_type == 'laplacian':
#         return fspecial_laplacian(*args, **kwargs)
#
#
# """
# # --------------------------------------------
# # degradation models
# # --------------------------------------------
# """
#
#
# def bicubic_degradation(x, sf=3):
#     '''
#     Args:
#         x: HxWxC image, [0, 1]
#         sf: down-scale factor
#
#     Return:
#         bicubicly downsampled LR image
#     '''
#     x = imresize_np(x, scale=1 / sf)
#     return x
#
#
# def srmd_degradation(x, k, sf=3):
#     ''' blur + bicubic downsampling
#
#     Args:
#         x: HxWxC image, [0, 1]
#         k: hxw, double
#         sf: down-scale factor
#
#     Return:
#         downsampled LR image
#
#     Reference:
#         @inproceedings{zhang2018learning,
#           title={Learning a single convolutional super-resolution network for multiple degradations},
#           author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
#           booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#           pages={3262--3271},
#           year={2018}
#         }
#     '''
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
#     x = bicubic_degradation(x, sf=sf)
#     return x
#
#
# def dpsr_degradation(x, k, sf=3):
#     ''' bicubic downsampling + blur
#
#     Args:
#         x: HxWxC image, [0, 1]
#         k: hxw, double
#         sf: down-scale factor
#
#     Return:
#         downsampled LR image
#
#     Reference:
#         @inproceedings{zhang2019deep,
#           title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
#           author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
#           booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#           pages={1671--1681},
#           year={2019}
#         }
#     '''
#     x = bicubic_degradation(x, sf=sf)
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
#     return x
#
#
# def classical_degradation(x, k, sf=3):
#     ''' blur + downsampling
#
#     Args:
#         x: HxWxC image, [0, 1]/[0, 255]
#         k: hxw, double
#         sf: down-scale factor
#
#     Return:
#         downsampled LR image
#     '''
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
#     # x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
#     st = 0
#     return x[st::sf, st::sf, ...]
#
#
# def add_sharpening(img, weight=0.5, radius=50, threshold=10):
#     """USM sharpening. borrowed from real-ESRGAN
#     Input image: I; Blurry image: B.
#     1. K = I + weight * (I - B)
#     2. Mask = 1 if abs(I - B) > threshold, else: 0
#     3. Blur mask:
#     4. Out = Mask * K + (1 - Mask) * I
#     Args:
#         img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
#         weight (float): Sharp weight. Default: 1.
#         radius (float): Kernel size of Gaussian blur. Default: 50.
#         threshold (int):
#     """
#     if radius % 2 == 0:
#         radius += 1
#     blur = cv2.GaussianBlur(img, (radius, radius), 0)
#     residual = img - blur
#     mask = np.abs(residual) * 255 > threshold
#     mask = mask.astype('float32')
#     soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
#
#     K = img + weight * residual
#     K = np.clip(K, 0, 1)
#     return soft_mask * K + (1 - soft_mask) * img
#
#
# def add_blur(img, sf=4):
#     wd2 = 4.0 + sf
#     wd = 2.0 + 0.2 * sf
#     if random.random() < 0.5:
#         l1 = wd2 * random.random()
#         l2 = wd2 * random.random()
#         k = anisotropic_Gaussian(ksize=2 * random.randint(2, 11) + 3, theta=random.random() * np.pi, l1=l1, l2=l2)
#     else:
#         k = fspecial('gaussian', 2 * random.randint(2, 11) + 3, wd * random.random())
#     img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
#
#     return img
#
#
# def add_resize(img, sf=4):
#     rnum = np.random.rand()
#     if rnum > 0.8:  # up
#         sf1 = random.uniform(1, 2)
#     elif rnum < 0.7:  # down
#         sf1 = random.uniform(0.5 / sf, 1)
#     else:
#         sf1 = 1.0
#     img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])), interpolation=random.choice([1, 2, 3]))
#     img = np.clip(img, 0.0, 1.0)
#
#     return img
#
#
# def add_Gaussian_noise(img, noise_level1=2, noise_level2=8):
#     noise_level = random.randint(noise_level1, noise_level2)
#     rnum = np.random.rand()
#     if rnum > 0.6:  # add color Gaussian noise
#         img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
#     elif rnum < 0.4:  # add grayscale Gaussian noise
#         img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
#     else:  # add  noise
#         L = noise_level2 / 255.
#         D = np.diag(np.random.rand(3))
#         U = orth(np.random.rand(3, 3))
#         conv = np.dot(np.dot(np.transpose(U), D), U)
#         img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
#     img = np.clip(img, 0.0, 1.0)
#     return img
#
#
# def add_speckle_noise(img, noise_level1=2, noise_level2=25):
#     noise_level = random.randint(noise_level1, noise_level2)
#     img = np.clip(img, 0.0, 1.0)
#     rnum = random.random()
#     if rnum > 0.6:
#         img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
#     elif rnum < 0.4:
#         img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
#     else:
#         L = noise_level2 / 255.
#         D = np.diag(np.random.rand(3))
#         U = orth(np.random.rand(3, 3))
#         conv = np.dot(np.dot(np.transpose(U), D), U)
#         img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
#     img = np.clip(img, 0.0, 1.0)
#     return img
#
#
# def add_Poisson_noise(img):
#     img = np.clip((img * 255.0).round(), 0, 255) / 255.
#     vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
#     if random.random() < 0.5:
#         img = np.random.poisson(img * vals).astype(np.float32) / vals
#     else:
#         img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
#         img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
#         noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
#         img += noise_gray[:, :, np.newaxis]
#     img = np.clip(img, 0.0, 1.0)
#     return img
#
#
# def poisson_degradation_pipeline(img, target_size=(64, 64)):
#     """
#     泊松噪声退化流程
#     Args:
#         img: 输入图像 (256x256)
#         target_size: 目标尺寸 (64x64)
#     Return:
#         退化后的64x64图像
#     """
#     # 1. 添加泊松噪声
#     noisy_img = add_Poisson_noise(img)
#
#     # 2. 下采样到目标尺寸
#     downsampled = cv2.resize(noisy_img, target_size[::-1], interpolation=cv2.INTER_CUBIC)
#
#     return downsampled
#
#
# def process_folder_poisson(input_dir, output_dir, target_size=(64, 64)):
#     """
#     处理整个文件夹的图像
#     Args:
#         input_dir: 输入图像目录
#         output_dir: 输出图像目录
#         target_size: 目标尺寸 (height, width)
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取图像
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 转换为RGB格式并归一化到[0,1]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_float = uint2single(img_rgb)
#
#         # 3. 应用泊松噪声退化流程
#         degraded_img = poisson_degradation_pipeline(img_float, target_size=target_size)
#
#         # 4. 转换回uint8并保存
#         output_img = single2uint(degraded_img)
#         output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
#
#         output_path = os.path.join(output_dir, f"{img_file}")
#         cv2.imwrite(output_path, output_img_bgr)
#
#     print("处理完成!")
#
#
# # def add_JPEG_noise(img):
# #     quality_factor = random.randint(30, 95)
# #     img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
# #     result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
# #     img = cv2.imdecode(encimg, 1)
# #     img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
# #     return img
# def add_JPEG_noise(img, quality_min=30, quality_max=95):
#     """
#     添加JPEG压缩噪声
#     Args:
#         img: [0,1]范围的图像
#         quality_min: 最低质量因子
#         quality_max: 最高质量因子
#     Return:
#         添加JPEG压缩伪影后的图像
#     """
#     # 转换为uint8
#     img_uint = single2uint(img)
#
#     # 转换为BGR格式（OpenCV要求）
#     img_bgr = cv2.cvtColor(img_uint, cv2.COLOR_RGB2BGR)
#
#     # 随机生成质量因子
#     quality_factor = np.random.randint(quality_min, quality_max + 1)
#
#     # JPEG编码
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
#     _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
#
#     # JPEG解码
#     img_decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
#
#     # 转换回RGB和[0,1]范围
#     img_rgb = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
#     return uint2single(img_rgb)
#
#
# def jpeg_degradation_pipeline(img, target_size=(64, 64), quality_min=30, quality_max=95):
#     """
#     JPEG退化流程
#     Args:
#         img: 输入图像 (256x256)
#         target_size: 目标尺寸 (64x64)
#         quality_min: JPEG最低质量
#         quality_max: JPEG最高质量
#     Return:
#         退化后的64x64图像
#     """
#     # 1. 初始JPEG压缩 (模拟原始压缩)
#     img = add_JPEG_noise(img, quality_min, quality_max)
#
#     # 2. 下采样到目标尺寸
#     img_down = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_CUBIC)
#
#     # 3. 再次JPEG压缩 (模拟传输/存储压缩)
#     img_down = add_JPEG_noise(img_down, quality_min, quality_max)
#
#     return img_down
#
#
# def process_folder_jpeg(input_dir, output_dir, target_size=(64, 64),
#                         quality_min=30, quality_max=95):
#     """
#     处理整个文件夹的图像
#     Args:
#         input_dir: 输入图像目录
#         output_dir: 输出图像目录
#         target_size: 目标尺寸 (height, width)
#         quality_min: JPEG最低质量
#         quality_max: JPEG最高质量
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#     print(f"JPEG质量范围: {quality_min}-{quality_max}")
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取图像
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 转换为RGB格式并归一化到[0,1]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_float = uint2single(img_rgb)
#
#         # 3. 应用JPEG退化流程
#         degraded_img = jpeg_degradation_pipeline(
#             img_float,
#             target_size=target_size,
#             quality_min=quality_min,
#             quality_max=quality_max
#         )
#
#         # 4. 转换回uint8并保存
#         output_img = single2uint(degraded_img)
#         output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
#
#         output_path = os.path.join(output_dir, f"{img_file}")
#         cv2.imwrite(output_path, output_img_bgr)
#
#     print("处理完成!")
#
#
# def random_crop(lq, hq, sf=4, lq_patchsize=64):
#     h, w = lq.shape[:2]
#     rnd_h = random.randint(0, h - lq_patchsize)
#     rnd_w = random.randint(0, w - lq_patchsize)
#     lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]
#
#     rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
#     hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf, :]
#     return lq, hq
#
#
# def degradation_bsrgan(img, sf=4, lq_patchsize=72, isp_model=None):
#     """
#     This is the degradation model of BSRGAN from the paper
#     "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
#     ----------
#     img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
#     sf: scale factor
#     isp_model: camera ISP model
#
#     Returns
#     -------
#     img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
#     hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
#     """
#     isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
#     sf_ori = sf
#
#     h1, w1 = img.shape[:2]
#     img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
#     h, w = img.shape[:2]
#
#     if h < lq_patchsize * sf or w < lq_patchsize * sf:
#         raise ValueError(f'img size ({h1}X{w1}) is too small!')
#
#     hq = img.copy()
#
#     if sf == 4 and random.random() < scale2_prob:  # downsample1
#         if np.random.rand() < 0.5:
#             img = cv2.resize(img, (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])),
#                              interpolation=random.choice([1, 2, 3]))
#         else:
#             img = imresize_np(img, 1 / 2, True)
#         img = np.clip(img, 0.0, 1.0)
#         sf = 2
#
#     shuffle_order = random.sample(range(7), 7)
#     idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
#     if idx1 > idx2:  # keep downsample3 last
#         shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]
#
#     for i in shuffle_order:
#
#         if i == 0:
#             img = add_blur(img, sf=sf)
#
#         elif i == 1:
#             img = add_blur(img, sf=sf)
#
#         elif i == 2:
#             a, b = img.shape[1], img.shape[0]
#             # downsample2
#             if random.random() < 0.75:
#                 sf1 = random.uniform(1, 2 * sf)
#                 img = cv2.resize(img, (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])),
#                                  interpolation=random.choice([1, 2, 3]))
#             else:
#                 k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
#                 k_shifted = shift_pixel(k, sf)
#                 k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
#                 img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
#                 img = img[0::sf, 0::sf, ...]  # nearest downsampling
#             img = np.clip(img, 0.0, 1.0)
#
#         elif i == 3:
#             # downsample3
#             img = cv2.resize(img, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
#             img = np.clip(img, 0.0, 1.0)
#
#         elif i == 4:
#             # add Gaussian noise
#             img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
#
#         elif i == 5:
#             # add JPEG noise
#             if random.random() < jpeg_prob:
#                 img = add_JPEG_noise(img)
#
#         elif i == 6:
#             # add processed camera sensor noise
#             if random.random() < isp_prob and isp_model is not None:
#                 with torch.no_grad():
#                     img, hq = isp_model.forward(img.copy(), hq)
#
#     # add final JPEG compression noise
#     img = add_JPEG_noise(img)
#
#     # random crop
#     img, hq = random_crop(img, hq, sf_ori, lq_patchsize)
#
#     return img, hq
#
#
# def degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.5, use_sharp=True, lq_patchsize=64, isp_model=None):
#     """
#     This is an extended degradation model by combining
#     the degradation models of BSRGAN and Real-ESRGAN
#     ----------
#     img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
#     sf: scale factor
#     use_shuffle: the degradation shuffle
#     use_sharp: sharpening the img
#
#     Returns
#     -------
#     img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
#     hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
#     """
#
#     h1, w1 = img.shape[:2]
#     img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
#     h, w = img.shape[:2]
#
#     if h < lq_patchsize * sf or w < lq_patchsize * sf:
#         raise ValueError(f'img size ({h1}X{w1}) is too small!')
#
#     if use_sharp:
#         img = add_sharpening(img)
#     hq = img.copy()
#
#     if random.random() < shuffle_prob:
#         shuffle_order = random.sample(range(13), 13)
#     else:
#         shuffle_order = list(range(13))
#         # local shuffle for noise, JPEG is always the last one
#         shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
#         shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))
#
#     poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1
#
#     for i in shuffle_order:
#         if i == 0:
#             img = add_blur(img, sf=sf)
#         elif i == 1:
#             img = add_resize(img, sf=sf)
#         elif i == 2:
#             img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
#         elif i == 3:
#             if random.random() < poisson_prob:
#                 img = add_Poisson_noise(img)
#         elif i == 4:
#             if random.random() < speckle_prob:
#                 img = add_speckle_noise(img)
#         elif i == 5:
#             if random.random() < isp_prob and isp_model is not None:
#                 with torch.no_grad():
#                     img, hq = isp_model.forward(img.copy(), hq)
#         elif i == 6:
#             img = add_JPEG_noise(img)
#         elif i == 7:
#             img = add_blur(img, sf=sf)
#         elif i == 8:
#             img = add_resize(img, sf=sf)
#         elif i == 9:
#             img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
#         elif i == 10:
#             if random.random() < poisson_prob:
#                 img = add_Poisson_noise(img)
#         elif i == 11:
#             if random.random() < speckle_prob:
#                 img = add_speckle_noise(img)
#         elif i == 12:
#             if random.random() < isp_prob and isp_model is not None:
#                 with torch.no_grad():
#                     img, hq = isp_model.forward(img.copy(), hq)
#         else:
#             print('check the shuffle!')
#
#     # resize to desired size
#     img = cv2.resize(img, (int(1 / sf * hq.shape[1]), int(1 / sf * hq.shape[0])),
#                      interpolation=random.choice([1, 2, 3]))
#
#     # add final JPEG compression noise
#     img = add_JPEG_noise(img)
#
#     # random crop
#     img, hq = random_crop(img, hq, sf, lq_patchsize)
#
#     return img, hq
#
#
# def process_folder_gaussian_noise(input_dir, output_dir, target_size=(64, 64), noise_level1 = 2,noise_level2=25):
#     """
#     处理整个文件夹的图像
#     Args:
#         input_dir: 输入图像目录
#         output_dir: 输出图像目录
#         target_size: 目标尺寸 (height, width)
#         noise_level: 高斯噪声强度 (0-100)
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取图像
#         img_path = os.path.join(input_dir, img_file)
#         img = imread_uint(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 转换为[0,1]范围
#         img_float = uint2single(img)
#
#         # 3. 添加高斯噪声
#         noisy_img = add_Gaussian_noise(img_float,  noise_level1, noise_level2)
#
#         # 4. 下采样到目标尺寸
#         downsampled = cv2.resize(noisy_img, (target_size[1], target_size[0]),
#                                  interpolation=cv2.INTER_CUBIC)
#
#         # 5. 转换回uint8并保存
#         output_img = single2uint(downsampled)
#         output_path = os.path.join(output_dir, f"{img_file}")
#         cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
#
#     print("处理完成!")
#
#
# def generate_gaussian_kernel(size=7, sigma=1.6):
#     """
#     生成高斯模糊核
#     Args:
#         size: 核大小 (奇数)
#         sigma: 高斯标准差
#     Return:
#         高斯模糊核
#     """
#     # 确保核大小为奇数
#     if size % 2 == 0:
#         size += 1
#
#     # 创建坐标网格
#     ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
#     xx, yy = np.meshgrid(ax, ax)
#
#     # 计算高斯分布
#     kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
#
#     # 归一化
#     return kernel / np.sum(kernel)
#
#
# def process_folder_dpsr(input_dir, output_dir, target_size=(64, 64), kernel_size=7, sigma=1.6):
#     """
#     处理整个文件夹的图像
#     Args:
#         input_dir: 输入图像目录
#         output_dir: 输出图像目录
#         target_size: 目标尺寸 (height, width)
#         kernel_size: 模糊核大小
#         sigma: 高斯模糊标准差
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 生成模糊核
#     kernel = generate_gaussian_kernel(kernel_size, sigma)
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
#
#     print(f"开始处理 {len(image_files)} 张图像...")
#     print(f"使用模糊核: {kernel_size}x{kernel_size}, σ={sigma}")
#
#     for img_file in tqdm(image_files, desc="处理进度"):
#         # 1. 读取图像
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#
#         # 2. 转换为RGB格式并归一化到[0,1]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_float = img_rgb.astype(np.float32) / 255.0
#
#         # 3. 应用DPSR退化
#         # 计算下采样因子 (256->64)
#         sf = img_float.shape[0] // target_size[0]
#         degraded_img = dpsr_degradation(img_float, kernel, sf=sf)
#
#         # 4. 确保输出尺寸正确
#         if degraded_img.shape[0] != target_size[0] or degraded_img.shape[1] != target_size[1]:
#             degraded_img = cv2.resize(degraded_img, (target_size[1], target_size[0]),
#                                       interpolation=cv2.INTER_CUBIC)
#
#         # 5. 转换回uint8并保存
#         output_img = (degraded_img * 255).clip(0, 255).astype(np.uint8)
#         output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
#
#         output_path = os.path.join(output_dir, f"{img_file}")
#         cv2.imwrite(output_path, output_img_bgr)
#
#     print("处理完成!")


# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_bosen"  # 输出目录
#
#     # 参数配置
#     TARGET_SIZE = (64, 64)  # 目标尺寸
#
#     # 执行处理
#     process_folder_poisson(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         target_size=TARGET_SIZE
#     )
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_JPEGweiying"  # 输出目录
#
#     # 参数配置
#     TARGET_SIZE = (64, 64)  # 目标尺寸
#     QUALITY_MIN = 99999  # JPEG最低质量
#     QUALITY_MAX = 100000 # JPEG最高质量
#
#     # 执行处理
#     process_folder_jpeg(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         target_size=TARGET_SIZE,
#         quality_min=QUALITY_MIN,
#         quality_max=QUALITY_MAX
#     )
# 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为原始图像目录
#     OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_hougaosimohu"  # 输出目录
#
#     # 参数配置
#     TARGET_SIZE = (64, 64)  # 目标尺寸
#     KERNEL_SIZE = 5  # 模糊核大小 (推荐5,7,9)
#     SIGMA = 0.4  # 高斯模糊标准差
#
#     # 执行处理
#     process_folder_dpsr(
#         INPUT_DIR,
#         OUTPUT_DIR,
#         target_size=TARGET_SIZE,
#         kernel_size=KERNEL_SIZE,
#         sigma=SIGMA
#     )

# if __name__ == "__main__":
#         INPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM"  # 替换为原始图像目录
#         OUTPUT_DIR = "/tmp/pycharm_project_ATD2/datasets/testUcM_gussionNoise2"  # 输出目录
#
#         # 参数配置
#         TARGET_SIZE = (64, 64)  # 目标尺寸
#         NOISE_LEVEL1 = 0
#         NOISE_LEVEL2 = 1  # 噪声水平 (0-100)
#
#         # 执行处理
#         process_folder_gaussian_noise(
#             INPUT_DIR,
#             OUTPUT_DIR,
#             target_size=TARGET_SIZE,
#             noise_level1=NOISE_LEVEL1,
#             noise_level2=NOISE_LEVEL2
#         )

# if __name__ == '__main__':
#     img = imread_uint('/tmp/pycharm_project_ATD2/datasets/testUcM/agricultural00.tif', 3)
#     img = uint2single(img)
#     sf = 4
#
#     for i in range(20):
#         img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=64)
#         print(i)
#         lq_nearest = cv2.resize(single2uint(img_lq), (int(sf * img_lq.shape[1]), int(sf * img_lq.shape[0])),
#                                 interpolation=0)
#         img_concat = np.concatenate([lq_nearest, single2uint(img_hq)], axis=1)
#         imsave(img_concat, str(i) + '.png')

#    for i in range(10):
#        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
#        print(i)
#        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#        util.imsave(img_concat, str(i)+'.png')

#    run utils/utils_blindsr.py