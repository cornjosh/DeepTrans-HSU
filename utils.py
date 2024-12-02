import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import numpy as np  # 导入NumPy库

# 计算均方根误差（RMSE）
def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape  # 获取输入图像的宽、高和通道数
    class_rmse = [0] * c  # 初始化每个通道的RMSE列表
    for i in range(c):  # 遍历每个通道
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))  # 计算每个通道的RMSE
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))  # 计算所有通道的平均RMSE
    return class_rmse, mean_rmse  # 返回每个通道的RMSE和平均RMSE

# 计算相对误差（RE）
def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape  # 获取输入图像的宽、高和通道数
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))  # 计算并返回RE

# 计算光谱角度距离（SAD）
def compute_sad(inp, target):
    p = inp.shape[-1]  # 获取输入的最后一个维度大小
    sad_err = [0] * p  # 初始化SAD误差列表
    for i in range(p):  # 遍历每个维度
        inp_norm = np.linalg.norm(inp[:, i], 2)  # 计算输入的L2范数
        tar_norm = np.linalg.norm(target[:, i], 2)  # 计算目标的L2范数
        summation = np.matmul(inp[:, i].T, target[:, i])  # 计算输入和目标的点积
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))  # 计算SAD误差
    mean_sad = np.mean(sad_err)  # 计算平均SAD误差
    return sad_err, mean_sad  # 返回每个维度的SAD误差和平均SAD误差

# 计算核范数
def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape  # 获取输入的形状
    inp = torch.reshape(inputs, (band, h * w))  # 重塑输入
    out = torch.norm(inp, p='nuc')  # 计算核范数
    return out  # 返回核范数

# 稀疏KL损失类
class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()  # 调用父类的构造函数

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)  # 对输入进行求和
        loss = Nuclear_norm(inp)  # 计算核范数损失
        return decay * loss  # 返回加权后的损失

# 和为一损失类
class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()  # 调用父类的构造函数
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))  # 注册常量张量
        self.loss = nn.L1Loss(reduction='sum')  # 使用L1损失

    def get_target_tensor(self, inp):
        target_tensor = self.one  # 获取目标张量
        return target_tensor.expand_as(inp)  # 扩展目标张量的形状

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)  # 对输入进行求和
        target_tensor = self.get_target_tensor(inp)  # 获取目标张量
        loss = self.loss(inp, target_tensor)  # 计算损失
        return gamma_reg * loss  # 返回加权后的损失

# SAD损失类
class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()  # 调用父类的构造函数
        self.num_bands = num_bands  # 设置波段数

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))  # 计算输入的范数
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))  # 计算目标的范数

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))  # 计算输入和目标的点积
            angle = torch.acos(summation / (input_norm * target_norm))  # 计算角度

        except ValueError:
            return 0.0  # 如果出现错误，返回0.0

        return angle  # 返回角度

# SID损失类
class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()  # 调用父类的构造函数
        self.eps = epsilon  # 设置epsilon值

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps  # 归一化输入
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps  # 归一化目标
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))  # 计算SID损失

        return sid  # 返回SID损失
