import torch
import torch.nn as nn
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small

class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        self.encoder = nn.Sequential(
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

        self.vtrans = ViT_small(image_size=size, patch_size=patch, num_classes=(dim*P), dim=(dim*P), depth=2,
                            heads=8, mlp_dim=12, channels=(dim*P)//patch**2, dropout=0.1, emb_dropout=0.1, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.P, -1)
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        return abu_est, re_result

    # 冻结/解冻 encoder
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    # 冻结/解冻 ViT
    def freeze_vit(self):
        for param in self.vtrans.parameters():
            param.requires_grad = False
    def unfreeze_vit(self):
        for param in self.vtrans.parameters():
            param.requires_grad = True

    # 冻结/解冻 upscale
    def freeze_upscale(self):
        for param in self.upscale.parameters():
            param.requires_grad = False
    def unfreeze_upscale(self):
        for param in self.upscale.parameters():
            param.requires_grad = True

    # 冻结/解冻 smooth
    def freeze_smooth(self):
        for param in self.smooth.parameters():
            param.requires_grad = False
    def unfreeze_smooth(self):
        for param in self.smooth.parameters():
            param.requires_grad = True

    # 冻结/解冻 decoder
    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    # 一键冻结/解冻所有参数
    def freeze_all(self):
        self.freeze_encoder()
        self.freeze_vit()
        self.freeze_upscale()
        self.freeze_smooth()
        self.freeze_decoder()
    def unfreeze_all(self):
        self.unfreeze_encoder()
        self.unfreeze_vit()
        self.unfreeze_upscale()
        self.unfreeze_smooth()
        self.unfreeze_decoder()


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)