import torch
import torch.nn as nn
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small

class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        
        # 1. Spatial Stream Encoder (CNN + ViT)
        self.spa_cnn = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        self.spa_vit = ViT_small(image_size=size, patch_size=patch, num_classes=(dim*P), dim=(dim*P), depth=2,
                            heads=8, mlp_dim=12, channels=(dim*P)//patch**2, dropout=0.1, emb_dropout=0.1, pool='cls')
        
        self.spa_upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )

        # 2. Spectral Stream Encoder (Fully Connected)
        self.spr_encoder = nn.Sequential(
            nn.Linear(L, 128),
            nn.ReLU(),
            nn.Linear(128, P),
            nn.ReLU()
        )
        
        # Smoothing layer for each abundance map
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        # 3. Decoder (fuses abundances)
        self.decoder = nn.Sequential(
            # Input channels are 2*P because we concatenate the two abundance maps
            nn.Conv2d(2 * P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        # 1. Spatial stream
        spa_feat = self.spa_cnn(x)
        cls_emb = self.spa_vit(spa_feat)
        cls_emb = cls_emb.view(1, self.P, -1)
        abu_spa = self.spa_upscale(cls_emb).view(1, self.P, self.size, self.size)
        
        # 2. Spectral stream
        # Reshape for FC layers: (N, C, H, W) -> (N*H*W, C)
        n, c, h, w = x.shape
        spr_in = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
        abu_spr = self.spr_encoder(spr_in)
        # Reshape back to image format: (N*H*W, P) -> (N, H, W, P) -> (N, P, H, W)
        abu_spr = abu_spr.view(n, h, w, self.P).permute(0, 3, 1, 2)

        # 3. Smooth both abundance maps before fusion
        abu_spa_s = self.smooth(abu_spa)
        abu_spr_s = self.smooth(abu_spr)

        # 4. Fusion and Decoder
        # Concatenate along the channel dimension
        abu_fused = torch.cat((abu_spa_s, abu_spr_s), dim=1)
        
        re_result = self.decoder(abu_fused)
        
        # Return the fused abundance and the reconstruction
        return abu_fused, re_result


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)