import random  # 导入随机数模块
import torch  # 导入PyTorch模块
import numpy as np  # 导入NumPy模块
import Trans_mod  # 导入自定义的Trans_mod模块

# 设置随机种子，以确保实验的可重复性
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 设备配置，优先使用GPU，如果GPU不可用则使用CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nSelected device:", device, end="\n\n")  # 输出选择的设备

# 创建Trans_mod模块中的Train_test对象，并传入相关参数
tmod = Trans_mod.Train_test(dataset='apex', device=device, skip_train=False, save=True)
tmod.run(smry=False)  # 运行训练和测试方法
