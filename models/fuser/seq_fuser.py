import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.registry import MODELS


@MODELS.register_module()
class SEQFuser(nn.Module):

    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.conv = nn.Sequential(
            # fuse
            ConvModule(128, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(256, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(512, 256, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):

        return self.conv(stack_feats)
