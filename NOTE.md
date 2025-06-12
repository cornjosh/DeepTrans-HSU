# 高光谱解混评估指标说明


## 1. RE (Relative Error) - 相对误差

### 定义
相对误差用于衡量重建图像与原始图像之间的差异，评估整体重建质量。

### 数学公式
```
RE = √[Σ(x_true - x_pred)² / (W × H × C)]
```

### 代码实现
```python
def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape  # 获取输入图像的宽、高和通道数
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))  # 计算并返回RE
```

### 物理意义
- **评估对象**: 重建高光谱图像 vs 原始高光谱图像
- **单位**: 与原始数据单位相同
- **含义**: 数值越小表示重建效果越好
- **应用**: 评估端到端系统性能，反映信息保存度和模型有效性

---

## 2. RMSE (Root Mean Squared Error) - 均方根误差

### 定义
均方根误差是MSE的平方根，用于评估丰度图估计的精度。

### 数学公式
```
RMSE = √[(1/n) × Σ(y_true - y_pred)²]
```

### 代码实现
```python
def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape  # 获取输入图像的宽、高和通道数
    class_rmse = [0] * c  # 初始化每个通道的RMSE列表
    for i in range(c):  # 遍历每个通道
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))  # 计算每个通道的RMSE
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))  # 计算所有通道的平均RMSE
    return class_rmse, mean_rmse  # 返回每个通道的RMSE和平均RMSE
```

### 物理意义
- **评估对象**: 估计丰度图 vs 真实丰度图
- **单位**: 与原始数据单位相同（比MSE更直观）
- **含义**: 数值越小表示丰度估计越准确
- **应用**: 评估每个端元的丰度估计精度，便于理解实际误差大小

---

## 3. SAD (Spectral Angle Distance) - 光谱角度距离

### 定义
光谱角度距离用于衡量两个光谱向量之间的角度差异，评估端元提取的精度。

### 数学公式
```
SAD = arccos(x₁ᵀx₂ / (||x₁|| × ||x₂||))
```

### 代码实现
```python
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
```

### 物理意义
- **评估对象**: 估计端元光谱 vs 真实端元光谱
- **单位**: 弧度（radians）
- **含义**: 数值越小表示光谱形状越相似，端元提取越准确
- **优势**: 对光谱强度变化不敏感，主要关注光谱形状的相似性

---

## 指标对比总结

| 指标 | 评估对象 | 单位 | 主要用途 | 特点 |
|------|----------|------|----------|------|
| **RE** | 重建图像 vs 原始图像 | 原始单位 | 整体重建质量 | 端到端性能评估 |
| **RMSE** | 丰度图 vs 真实丰度 | 原始单位 | 丰度估计精度 | 比MSE更直观 |
| **SAD** | 端元光谱 vs 真实端元 | 弧度 | 端元提取精度 | 关注光谱形状相似性 |

## 在训练中的应用

### 损失函数（训练阶段）
```python
loss_re = self.beta * loss_func(re_result, x)  # MSE重建损失
loss_sad = self.gamma * torch.sum(loss_sad).float()  # SAD光谱角度损失
total_loss = loss_re + loss_sad  # 总损失
```

### 评估指标（测试阶段）
```python
re = utils.compute_re(x, re_result)  # 重建误差
rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)  # 丰度误差
sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)  # 端元误差
```

---

## 端元光谱特征曲线和丰度图的来源

### 1. 端元光谱特征曲线的来源

端元光谱特征存储在**Decoder的权重参数**中：

```python
# 在AutoEncoder的decoder定义中
self.decoder = nn.Sequential(
    nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),  # P个端元 -> L个波段
    nn.ReLU(),
)
```

#### 提取过程：
```python
# 从训练好的模型中提取端元光谱
est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()  # 获取decoder第一层权重
est_endmem = est_endmem.reshape((self.L, self.P))  # 重塑为 (波段数L, 端元数P)
est_endmem = est_endmem[:, self.order_endmem]  # 调整端元顺序
```

#### 物理意义：
- **维度**: `(L, P)` - L个波段，P个端元
- **含义**: 每一列代表一个端元在所有波段上的反射率值
- **学习方式**: 通过重建损失反向传播自动学习得到
- **约束**: 通过`NonZeroClipper`确保权重在[1e-6, 1]范围内

### 2. 丰度图的来源

丰度图通过**Encoder + ViT + Upscale + Smooth**流水线生成：

#### 生成流程：
```python
def forward(self, x):
    # 1. Encoder: 光谱特征提取
    abu_est = self.encoder(x)  # (1, L, H, W) -> (1, feature_dim, H, W)
    
    # 2. ViT: 空间上下文学习
    cls_emb = self.vtrans(abu_est)  # 学习空间依赖关系
    
    # 3. Upscale: 特征到空间映射
    cls_emb = cls_emb.view(1, self.P, -1)
    abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
    
    # 4. Smooth: 空间平滑和归一化
    abu_est = self.smooth(abu_est)  # 3x3卷积 + Softmax归一化
    
    return abu_est, re_result
```

#### 各模块作用：

1. **Encoder (编码器)**：
   - 输入：高光谱图像 `(1, L, H, W)`
   - 作用：提取每个像素的光谱特征
   - 输出：特征图 `(1, feature_dim, H, W)`

2. **ViT (Vision Transformer)**：
   - 输入：编码后的特征图
   - 作用：学习像素间的空间上下文关系
   - 输出：全局特征向量

3. **Upscale (上采样)**：
   - 输入：ViT的全局特征
   - 作用：将特征映射回原始空间尺寸
   - 输出：初始丰度图 `(1, P, H, W)`

4. **Smooth (平滑层)**：
   - 输入：初始丰度图
   - 作用：空间平滑 + Softmax归一化
   - 输出：最终丰度图 `(1, P, H, W)`

#### 最终处理：
```python
# 测试阶段的后处理
abu_est = abu_est / (torch.sum(abu_est, dim=1))  # 确保每个像素丰度和为1
abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # 转换为(H, W, P)
abu_est = abu_est[:, :, self.order_abd]  # 调整端元顺序
```

#### 物理意义：
- **维度**: `(H, W, P)` - 高度H，宽度W，P个端元
- **含义**: 每个像素位置(i,j)处，P个端元的混合比例
- **约束**: 每个像素的丰度和为1（和为一约束）
- **范围**: [0, 1]之间的概率值

### 3. 线性混合模型

端元和丰度图通过线性混合模型重建原始图像：

```python
# 重建过程（在decoder中实现）
re_result = self.decoder(abu_est)  # 丰度图 × 端元光谱 = 重建图像
```

#### 数学表达：
```
重建像素(i,j,k) = Σ(p=1 to P) [丰度(i,j,p) × 端元光谱(k,p)]
```

其中：
- `i,j`: 空间位置坐标
- `k`: 光谱波段索引
- `p`: 端元索引
- `P`: 端元总数

这种设计实现了**端到端的无监督学习**，通过重建高光谱图像同时学习端元光谱特征和像素丰度分布。