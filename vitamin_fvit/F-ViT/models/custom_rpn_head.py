import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RPNHead


@HEADS.register_module()
class CustomRPNHead(RPNHead):
    def __init__(self,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        self.norm_cfg = norm_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False,
                        norm_cfg=self.norm_cfg
                    ))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)
