import os  # 导入操作系统模块
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import time  # 导入时间模块

import scipy.io as sio  # 导入scipy.io模块，用于处理MAT文件
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torchsummary import summary  # 导入torchsummary模块，用于打印模型摘要

import datasets  # 导入自定义的datasets模块
import plots  # 导入自定义的plots模块
import transformer  # 导入自定义的transformer模块
import utils  # 导入自定义的utils模块


class AutoEncoder(nn.Module):  # 定义AutoEncoder类，继承自nn.Module
    def __init__(self, P, L, size, patch, dim):  # 初始化函数
        super(AutoEncoder, self).__init__()  # 调用父类的初始化函数
        self.P, self.L, self.size, self.dim = P, L, size, dim  # 初始化类属性
        self.encoder = nn.Sequential(  # 定义编码器部分
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 2D卷积层
            nn.BatchNorm2d(128, momentum=0.9),  # 批量归一化层
            nn.Dropout(0.25),  # Dropout层
            nn.LeakyReLU(),  # LeakyReLU激活函数
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 2D卷积层
            nn.BatchNorm2d(64, momentum=0.9),  # 批量归一化层
            nn.LeakyReLU(),  # LeakyReLU激活函数
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # 2D卷积层
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),  # 批量归一化层
        )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')  # 定义视觉Transformer部分
        
        self.upscale = nn.Sequential(  # 定义上采样部分
            nn.Linear(dim, size ** 2),  # 线性层
        )
        
        self.smooth = nn.Sequential(  # 定义平滑部分
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 2D卷积层
            nn.Softmax(dim=1),  # Softmax激活函数
        )

        self.decoder = nn.Sequential(  # 定义解码器部分
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),  # 2D卷积层
            nn.ReLU(),  # ReLU激活函数
        )

    @staticmethod
    def weights_init(m):  # 定义权重初始化函数
        if type(m) == nn.Conv2d:  # 如果层是2D��积层
            nn.init.kaiming_normal_(m.weight.data)  # 使用Kaiming正态分布初始化权重

    def forward(self, x):  # 定义前向传播函数
        abu_est = self.encoder(x)  # ��码输入
        cls_emb = self.vtrans(abu_est)  # 通过视觉Transformer处理
        cls_emb = cls_emb.view(1, self.P, -1)  # 调整形状
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)  # 上采样并调整形状
        abu_est = self.smooth(abu_est)  # 平滑处理
        re_result = self.decoder(abu_est)  # 解码
        return abu_est, re_result  # 返回估计的丰度图和重建结果


class NonZeroClipper(object):  # 定义NonZeroClipper类
    def __call__(self, module):  # 定义调用函数
        if hasattr(module, 'weight'):  # 如果模块有权重属性
            w = module.weight.data  # 获取权重数据
            w.clamp_(1e-6, 1)  # 将权重限制在1e-6到1之间


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

            self.LR, self.EPOCH = 6e-3, 200  # 学习率和训练轮数
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
            self.patch, self.dim = 5, 200  # patch大小和维度
            self.beta, self.gamma = 5e3, 1e-4  # 损失函数的权重
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
            summary(net, input_size=(self.L, self.col, self.col))  # 打印模型摘要
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
        
        if not self.skip_train:  # 如果不跳过训练
            time_start = time.time()  # 记录开始时间
            net.train()  # 设置模型为训练模式
            epo_vs_los = []  # 初始化损失记录列表
            for epoch in range(self.EPOCH):  # 训练循环
                for i, (x, _) in enumerate(self.loader):  # 遍历数据加载器

                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)  # 调整输入形状
                    abu_est, re_result = net(x)  # 前向传播

                    loss_re = self.beta * loss_func(re_result, x)  # 计算重建损失
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                          x.view(1, self.L, -1).transpose(1, 2))  # 计算光谱角度距离损失
                    loss_sad = self.gamma * torch.sum(loss_sad).float()  # 加权光谱角度距离损失

                    total_loss = loss_re + loss_sad  # 总损失

                    optimizer.zero_grad()  # 清空梯度
                    total_loss.backward()  # 反向传播
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)  # 梯度裁剪
                    optimizer.step()  # 优化器更新

                    net.decoder.apply(apply_clamp_inst1)  # 应用NonZeroClipper
                    
                    if epoch % 10 == 0:  # 每10个epoch打印一次损失
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))  # 记录损失

                scheduler.step()  # 更新学习率
            time_end = time.time()  # 记录结束时间
            
            if self.save:  # 如果需要保存模型
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:  # 打开文件
                    pickle.dump(net.state_dict(), handle)  # 保存模型状态字典
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})  # 保存损失记录
            
            print('Total computational cost:', time_end - time_start)  # 打印总计算时间

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
