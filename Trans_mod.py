import os  # 导入操作系统模块
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import time  # 导入时间模块

import numpy as np  # 用于数组变换
import scipy.io as sio  # 导入scipy.io模块，用于处理MAT文件
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torchinfo import summary  # 导入torchsummary模块，用于打印模型摘要

from vit_pytorch import ViT  # 从vit_pytorch库中导入ViT模块
from vit_pytorch import SimpleViT  # 从vit_pytorch库中导入SimpleViT模块
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small  # 从vit_pytorch库中导入ViT_small模块

import datasets  # 导入自定义的datasets模块
import plots  # 导入自定义的plots模块
import utils  # 导入自定义的utils模块
from autoencoder import AutoEncoder, NonZeroClipper


class Train_test:  # 定义Train_test类
    def __init__(self, dataset, device, skip_train=False, save=False):  # 初始化函数
        super(Train_test, self).__init__()  # 调用父类的初始化函数
        self.skip_train = skip_train  # 是否跳过训练
        self.device = device  # 设备
        self.dataset = dataset  # 数据集名称
        self.save = save  # 是否保存模型
        self.save_dir = "trans_mod_" + dataset + "/"  # 保存目录
        os.makedirs(self.save_dir, exist_ok=True)  # 创建保存目录
        if dataset == 'samson':  # 如果数据集是samson
            self.data = datasets.Data(dataset, device)  # 加载数据
            self.P, self.L, self.col = self.data.get_P(), self.data.get_L(), self.data.get_col()  # 初始化参数
            self.loader = self.data.get_loader(batch_size=self.col ** 2)  # 获取数据加载器
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()  # 初始化权重
            self.LR, self.EPOCH = 6e-3, 500  # 学习率和训练轮数
            self.stage1_epochs = 250
            self.stage2_epochs = 250
            self.patch, self.dim = 5, 200  # patch大小和维度
            self.beta, self.gamma = 5e3, 3e-2  # 损失函数的权重
            self.weight_decay_param = 4e-5  # 权重衰减参数
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)  # 丰度图和端元的顺序
        elif dataset == 'apex':  # 如果数据集是apex
            self.data = datasets.Data(dataset, device)  # 加载数据
            self.P, self.L, self.col = self.data.get_P(), self.data.get_L(), self.data.get_col()  # 初始化参数
            self.loader = self.data.get_loader(batch_size=self.col ** 2)  # 获取数据加载器
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()  # 初始化权重

            self.LR, self.EPOCH = 9e-3, 200  # 学习率和训练轮数
            self.stage1_epochs = 100
            self.stage2_epochs = 100
            self.patch, self.dim = 5, 200  # patch大小和维度
            self.beta, self.gamma = 5e3, 5e-2  # 损失函数的权重
            self.weight_decay_param = 4e-5  # 权重衰减参数
            self.order_abd, self.order_endmem = (3, 1, 2, 0), (3, 1, 2, 0)  # 丰度图和端元的顺序
        elif dataset == 'dc':  # 如果数据集是dc
            self.data = datasets.Data(dataset, device)  # 加载数据
            self.P, self.L, self.col = self.data.get_P(), self.data.get_L(), self.data.get_col()  # 初始化参数
            self.loader = self.data.get_loader(batch_size=self.col ** 2)  # 获取数据加载器
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()  # 初始化权重

            self.LR, self.EPOCH = 6e-3, 150  # 学习率和训练轮数
            self.stage1_epochs = 75
            self.stage2_epochs = 75
            self.patch, self.dim = 10, 400  # patch大小和维度
            self.beta, self.gamma = 5e3, 1e-4  # 损失函数的权重
            self.weight_decay_param = 3e-5  # 权重衰减参数
            self.order_abd, self.order_endmem = (0, 2, 1, 5, 4, 3), (0, 2, 1, 5, 4, 3)  # 丰度图和端元的顺序
        elif dataset == 'urban':  # 如果数据集是urban
            self.data = datasets.Data(dataset, device)  # 加载数据
            self.P, self.L, self.col = self.data.get_P(), self.data.get_L(), self.data.get_col()  # 初始化参数
            self.loader = self.data.get_loader(batch_size=self.col ** 2)  # 获取数据加载器
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()  # 初始化权重

            self.LR, self.EPOCH = 6e-3, 150  # 学习率和训练轮数
            self.stage1_epochs = 75
            self.stage2_epochs = 75
            self.patch, self.dim = 5, 200  # patch大小和维度
            self.beta, self.gamma = 5e3, 1e-4  # 损失函数的权重
            self.weight_decay_param = 3e-5  # 权重衰减参数
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3)  # 丰度图和端元的顺序
        elif dataset == 'jasper':  # 如果数据集是jasper
            self.data = datasets.Data(dataset, device)  # 加载数据
            self.P, self.L, self.col = self.data.get_P(), self.data.get_L(), self.data.get_col()  # 初始化参数
            self.loader = self.data.get_loader(batch_size=self.col ** 2)  # 获取数据加载器
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()  # 初始化权重

            self.LR, self.EPOCH = 6e-3, 150  # 学习率和训练轮数
            self.stage1_epochs = 75
            self.stage2_epochs = 75
            self.patch, self.dim = 5, 200  # patch大小和维度
            self.beta, self.gamma = 1e3, 5e-2  # 损失函数的权重
            self.weight_decay_param = 3e-5  # 权重衰减参数
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3)  # 丰度图和端元的顺序
        else:  # 如果数据集未知
            raise ValueError("Unknown dataset")  # 抛出异常

    def run(self, smry):  # 定义运行函数
        net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim).to(self.device)  # 初始化AutoEncoder模型
        if smry:  # 如果需要打印模型摘要
            # summary(net, (1, self.L, self.col, self.col), batch_dim=None)  # 打印模型摘要
            # print(net)  # 打印模型
            summary(net, input_size=(1, self.L, self.col, self.col))  # 打印模型摘要
            return

        net.apply(net.weights_init)  # 初始化模型权重

        model_dict = net.state_dict()  # 获取模型状态字典
        model_dict['decoder.0.weight'] = self.init_weight  # 设置解码器的初始权重
        net.load_state_dict(model_dict)  # 加载模型状态字典

        loss_func = nn.MSELoss(reduction='mean')  # 定义均方误差损失函数
        loss_func2 = utils.SAD(self.L)  # 定义光谱角度距离损失函数
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)  # 定义Adam优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)  # 定义学习率调度器
        apply_clamp_inst1 = NonZeroClipper()  # 定义NonZeroClipper实例
        
        # ==================== 两步训练 ====================
        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            # stage1_epochs = self.EPOCH // 2 # 使用 __init__ 中定义的
            # stage2_epochs = self.EPOCH - stage1_epochs # 使用 __init__ 中定义的

            # --------- 阶段1：增大SAD权重，训练全部参数 ---------
            print("阶段1：增大SAD权重，训练全部参数")
            net.unfreeze_all()
            sad_weight1 = self.gamma * 10  # 增大SAD权重
            mse_weight1 = self.beta * 0.1  # 降低MSE权重
            optimizer1 = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
            scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=15, gamma=0.8)
            for epoch in range(self.stage1_epochs):
                for i, (x, _) in enumerate(self.loader):
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)
                    loss_re = mse_weight1 * loss_func(re_result, x)
                    loss_sad = sad_weight1 * torch.sum(loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                                                x.view(1, self.L, -1).transpose(1, 2))).float()
                    total_loss = loss_re + loss_sad
                    optimizer1.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer1.step()                    
                    net.decoder.apply(apply_clamp_inst1)
                    if epoch % 10 == 0 and i == 0:
                        # 计算RMSE（只取当前batch，和测试阶段一致，需转为numpy）
                        with torch.no_grad():
                            abu_est_np = abu_est.detach().cpu().numpy()  # (1, P, col, col)
                            if hasattr(self.data, 'get') and callable(self.data.get):
                                target_np = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
                            else:
                                target_np = None
                            # 转换 abu_est 为 (col, col, P)
                            if abu_est_np.shape[0] == 1:
                                abu_est_show = np.moveaxis(abu_est_np.squeeze(0), 0, -1)
                            else:
                                abu_est_show = np.moveaxis(abu_est_np, 0, -1)
                            if target_np is not None and abu_est_show.shape == target_np.shape:
                                _, rmse_val = utils.compute_rmse(target_np, abu_est_show)
                            else:
                                rmse_val = -1
                        # 端元SAD统计
                        with torch.no_grad():
                            est_endmem = net.decoder[0].weight.detach().cpu().numpy().reshape(self.L, self.P)
                            true_endmem = self.data.get("end_mem").cpu().numpy()
                            est_endmem = est_endmem[:, self.order_endmem] if hasattr(self, 'order_endmem') else est_endmem
                            true_endmem = true_endmem[:, self.order_endmem] if hasattr(self, 'order_endmem') else true_endmem
                            _, mean_sad = utils.compute_sad(est_endmem, true_endmem)
                        print(f'[Stage1] Epoch: {epoch} | loss: {total_loss.item():.4f} | loss re: {loss_re.item():.4f} | loss SAD: {loss_sad.item():.4f} | true rmse: {rmse_val:.4f} | true SAD: {mean_sad:.4f}')
                    epo_vs_los.append(float(total_loss.item()))
                scheduler1.step()

            # --------- 阶段2：冻结decoder，增大MSE权重，训练丰度相关参数 ---------
            print("阶段2：冻结decoder，仅训练丰度相关参数，增大MSE权重")
            net.freeze_decoder()
            net.unfreeze_encoder()
            net.unfreeze_vit()
            net.unfreeze_upscale()
            net.unfreeze_smooth()
            sad_weight2 = self.gamma * 0.1  # 降低SAD权重
            mse_weight2 = self.beta * 10   # 增大MSE权重
            # 只优化未冻结参数
            params2 = filter(lambda p: p.requires_grad, net.parameters())
            optimizer2 = torch.optim.Adam(params2, lr=self.LR * 0.5, weight_decay=self.weight_decay_param)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=15, gamma=0.8)
            for epoch in range(self.stage2_epochs):
                for i, (x, _) in enumerate(self.loader):
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)
                    loss_re = mse_weight2 * loss_func(re_result, x)
                    loss_sad = sad_weight2 * torch.sum(loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                                                x.view(1, self.L, -1).transpose(1, 2))).float()
                    total_loss = loss_re + loss_sad
                    optimizer2.zero_grad()
                    total_loss.backward()                    
                    nn.utils.clip_grad_norm_(params2, max_norm=10, norm_type=1)
                    optimizer2.step()
                    if epoch % 10 == 0 and i == 0:
                        # 计算RMSE（只取当前batch，和测试阶段一致，需转为numpy）
                        with torch.no_grad():
                            abu_est_np = abu_est.detach().cpu().numpy()
                            x_np = x.detach().cpu().numpy()
                            if hasattr(self.data, 'get') and callable(self.data.get):
                                target_np = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
                            else:
                                target_np = None
                            # 转换 abu_est 为 (col, col, P)
                            if abu_est_np.shape[0] == 1:
                                abu_est_show = np.moveaxis(abu_est_np.squeeze(0), 0, -1)
                            else:
                                abu_est_show = np.moveaxis(abu_est_np, 0, -1)
                            if target_np is not None and abu_est_show.shape == target_np.shape:
                                _, rmse_val = utils.compute_rmse(target_np, abu_est_show)
                            else:
                                rmse_val = -1
                        # 端元SAD统计
                        with torch.no_grad():
                            est_endmem = net.decoder[0].weight.detach().cpu().numpy().reshape(self.L, self.P)
                            true_endmem = self.data.get("end_mem").cpu().numpy()
                            est_endmem = est_endmem[:, self.order_endmem] if hasattr(self, 'order_endmem') else est_endmem
                            true_endmem = true_endmem[:, self.order_endmem] if hasattr(self, 'order_endmem') else true_endmem
                            _, mean_sad = utils.compute_sad(est_endmem, true_endmem)
                        print(f'[Stage2] Epoch: {epoch} | loss: {total_loss.item():.4f} | loss re: {loss_re.item():.4f} | loss SAD: {loss_sad.item():.4f} | true rmse: {rmse_val:.4f} | true SAD: {mean_sad:.4f}')
                    epo_vs_los.append(float(total_loss.item()))
                scheduler2.step()

            time_end = time.time()
            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
            print('Total computational cost:', time_end - time_start)
        # ...existing code...

        else:  # 如果跳过训练
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:  # 打开文件
                net.load_state_dict(pickle.load(handle))  # 加载模型状态字典

        # 测试部分 ================

        net.eval()  # 设置模型为评估模式
        x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)  # 获取高光谱图像并调整形状
        abu_est, re_result = net(x)  # 前向传播
        abu_est = abu_est / (torch.sum(abu_est, dim=1))  # 归一化丰度图
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # 调整形状并转换为numpy数组
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()  # 获取目标丰度图
        true_endmem = self.data.get("end_mem").numpy()  # 获取真实端元
        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()  # 获取估计端元
        est_endmem = est_endmem.reshape((self.L, self.P))  # 调整形状

        abu_est = abu_est[:, :, self.order_abd]  # 调整丰度图顺序
        est_endmem = est_endmem[:, self.order_endmem]  # 调整端元顺序

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})  # 保存估计丰度图
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})  # 保存估计端元

        x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()  # 调整输入形状并转换为numpy数组
        re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()  # 调整重建结果形状并转换为numpy数组
        re = utils.compute_re(x, re_result)  # 计算重建误差
        print("RE:", re)  # 打印重建误差

        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)  # 计算均方根误差
        print("Class-wise RMSE value:")  # 打印每类的均方根误差
        for i in range(self.P):  # 遍历每类
            print("Class", i + 1, ":", rmse_cls[i])  # 打印每类的均方根误差
        print("Mean RMSE:", mean_rmse)  # 打印平均均方根误差

        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)  # 计算光谱角度距离
        print("Class-wise SAD value:")  # 打印每类的光谱角度距离
        for i in range(self.P):  # 遍历每类
            print("Class", i + 1, ":", sad_cls[i])  # 打印每类的光谱角度距离
        print("Mean SAD:", mean_sad)  # 打印平均光谱角度距离

        with open(self.save_dir + "log1.csv", 'a') as file:  # 打开日志文件
            file.write(f"LR: {self.LR}, ")  # 写入学习率
            file.write(f"WD: {self.weight_decay_param}, ")  # 写入权重衰减参数
            file.write(f"RE: {re:.4f}, ")  # 写入重建误差
            file.write(f"SAD: {mean_sad:.4f}, ")  # 写入平均光谱角度距离
            file.write(f"RMSE: {mean_rmse:.4f}\n")  # 写入平均均方根误差

        plots.plot_abundance(target, abu_est, self.P, self.save_dir)  # 绘制丰度图
        plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)  # 绘制端元图
        
# =================================================================

if __name__ == '__main__':  # 主函数入口
    pass  # 不执行任何操作
