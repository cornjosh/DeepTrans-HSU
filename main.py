import random  # 导入随机数模块
import torch  # 导入PyTorch模块
import numpy as np  # 导入NumPy模块
import Trans_mod  # 导入自定义的Trans_mod模块
import argparse  # 导入argparse模块

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Train and test model')
parser.add_argument('dataset', type=str, nargs='?', help='Dataset name')
parser.add_argument('--skip_train', action='store_true', help='Skip training')
parser.add_argument('--save', action='store_true', help='Save the model')
parser.add_argument('--smry', action='store_true', help='Save the model')
args = parser.parse_args()

if args.dataset is None:
    args.dataset = 'samson'  # 默认数据集为 samson

print("\nSelected dataset:", args.dataset)  # 输出选择的数据集
print("Skip training:", args.skip_train)  # 输出是否跳过训练
print("Save the model:", args.save)  # 输出是否保存模型

# 设置随机种子，以确保实验的可重复性
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 设备配置，优先使用GPU，如果GPU不可用则使用CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Selected device:", device, end="\n\n")  # 输出选择的设备

# 创建Trans_mod模块中的Train_test对象，并传入相关参数
tmod = Trans_mod.Train_test(dataset=args.dataset, device=device, skip_train=args.skip_train, save=args.save)
tmod.run(smry=args.smry)  # 运行训练和测试方法
