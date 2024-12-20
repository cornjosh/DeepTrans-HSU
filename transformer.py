import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入神经网络模块
from timm.models.layers import DropPath  # 从timm库中导入DropPath模块
from timm.models.vision_transformer import Mlp  # 从timm库中导入Mlp模块

from einops import rearrange, repeat  # 从einops库中导入rearrange和repeat函数
from einops.layers.torch import Rearrange  # 从einops库中导入Rearrange模块


# 辅助函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # 如果t是元组则返回t，否则返回(t, t)

# 类定义
class PreNorm(nn.Module):  # 定义PreNorm类，继承自nn.Module
    def __init__(self, dim, fn):
        super().__init__()  # 调用父类的初始化方法
        self.norm = nn.LayerNorm(dim)  # 定义LayerNorm层
        self.fn = fn  # 保存传入的函数

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 先进行LayerNorm，再调用传入的函数

class FeedForward(nn.Module):  # 定义FeedForward类，继承自nn.Module
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()  # 调用父类的初始化方法
        self.net = nn.Sequential(  # 定义前馈神经网络
            nn.Linear(dim, hidden_dim),  # 线性层
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

class CrossAttention(nn.Module):  # 定义CrossAttention类，继承自nn.Module
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()  # 调用父类的初始化方法
        self.num_heads = num_heads  # 头的数量
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"  # 确保dim能被num_heads整除
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)  # 定义查询向量的线性层
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)  # 定义键向量的线性层
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)  # 定义值向量的线性层
        self.attn_drop = nn.Dropout(attn_drop)  # 定义注意力的Dropout层
        self.proj = nn.Linear(dim, dim)  # 定义输出的线性层
        self.proj_drop = nn.Dropout(proj_drop)  # 定义输出的Dropout层

    def forward(self, x):
        B, N, C = x.shape  # 获取输入的形状
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 计算查询向量
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 计算键向量
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 计算值向量

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力权重
        attn = attn.softmax(dim=-1)  # 对注意力权重进行softmax
        attn = self.attn_drop(attn)  # 对注意力权重进行Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # 计算注意力输出
        x = self.proj(x)  # 通过线性层
        x = self.proj_drop(x)  # 通过Dropout层
        return x  # 返回输出

class CrossAttentionBlock(nn.Module):  # 定义CrossAttentionBlock类，继承自nn.Module
    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()  # 调用父类的初始化方法
        self.norm1 = norm_layer(dim)  # 定义第一个LayerNorm层
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)  # 定义CrossAttention层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 定义DropPath层
        self.has_mlp = has_mlp  # 是否包含MLP层
        if has_mlp:
            self.norm2 = norm_layer(dim)  # 定义第二个LayerNorm层
            mlp_hidden_dim = int(dim * mlp_ratio)  # 计算MLP的隐藏层维度
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 定义MLP层

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(x))  # 计算CrossAttention输出并加上输入
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # 计算MLP输出并加上输入
        return x  # 返回输出

class Transformer(nn.Module):  # 定义Transformer类，继承自nn.Module
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()  # 调用父类的初始化方法
        self.norm = nn.LayerNorm(dim)  # 定义LayerNorm层
        self.layers = nn.ModuleList([])  # 定义层列表
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # 添加CrossAttentionBlock和FeedForward层
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = torch.cat((attn(x), self.norm(x[:, 1:, :])), dim=1)  # 计算CrossAttentionBlock输出并拼接
            x = ff(x) + x  # 计算FeedForward输出并加上输入
        return x  # 返回输出

class ViT(nn.Module):  # 定义ViT类，继承自nn.Module
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()  # 调用父类的初始化方法
        image_height, image_width = pair(image_size)  # 获取图像高度和宽度
        patch_height, patch_width = pair(patch_size)  # 获取patch高度和宽度

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 确保图像尺寸能被patch尺寸整除

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 计算patch数量
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 确保pool类型为cls或mean

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 将图像重排为patch
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 定义位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 定义分类token
        self.dropout = nn.Dropout(emb_dropout)  # 定义Dropout层

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # 定义Transformer层

        self.pool = pool  # 定义pool类型
        self.to_latent = nn.Identity()  # 定义Identity层

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 计算patch嵌入
        b, n, _ = x.shape  # 获取输入的形状

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 重复分类token
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接分类token和patch嵌入
        x += self.pos_embedding[:, :(n + 1)]  # 加上位置嵌入
        x = self.dropout(x)  # 通过Dropout层
        
        x = self.transformer(x)  # 通过Transformer层

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # 进行池化
        x = self.to_latent(x)  # 通过Identity层
        
        return x  # 返回输出
