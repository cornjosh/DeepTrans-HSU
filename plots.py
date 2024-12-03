import numpy as np
from matplotlib import pyplot as plt

# 绘制丰度图
def plot_abundance(ground_truth, estimated, em, save_dir):
    # 创建一个新的图像，大小为12x6英寸，分辨率为150dpi
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        # 绘制真实丰度图
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')

    for i in range(em):
        # 绘制估计丰度图
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
    plt.tight_layout()

    # 保存图像到指定目录
    plt.savefig(save_dir + "abundance.png")

# 绘制端元图
def plot_endmembers(target, pred, em, save_dir):
    # 创建一个新的图像，大小为12x6英寸，分辨率为150dpi
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        # 绘制端元曲线
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i], label="Extracted")
        plt.plot(target[:, i], label="GT")
        plt.legend(loc="upper left")
    plt.tight_layout()

    # 保存图像到指定目录
    plt.savefig(save_dir + "end_members.png")
