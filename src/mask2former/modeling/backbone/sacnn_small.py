import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from utils.log import getLogger

logger = getLogger('unsup-vidseg.SA-CNN')



class SoftPositionEmbed(nn.Module):
    """Taken from Slot Attention"""
    def __init__(self, hsize, resolution):
        super(SoftPositionEmbed, self).__init__()

        h, w = resolution[-2:]
        hs = torch.linspace(0., 1., h)
        ws = torch.linspace(0., 1., w)
        c = torch.stack(torch.meshgrid(hs, ws), dim=0)
        grid = torch.cat([c, 1-c], dim=0)[None]

        self.register_buffer('grid', grid)
        self.aff = nn.Conv2d(4, hsize, 1, bias=True)

    def forward(self, input):
        return input + self.aff(self.grid)



class SlotAttentionCNNBackboneSMALL(nn.Module):
    """
    6-layer CNN backbone taken from Slot Attention by Locatelo et al.

    Note unlike the original implementation, here first convolution has stride 2 to lower the resolution of the input.
    """
    def __init__(self, input_shape, stride=2):
        super(SlotAttentionCNNBackboneSMALL, self).__init__()
        logger.info(f"Slot Attention CNN backbone input_shape {input_shape}")
        ic, h, w = input_shape[-3:]
        # Split into 4 stages that Mask2Former
        self.encoder_cnn_0 = nn.Sequential(
            nn.Conv2d(ic, 32, kernel_size=5, stride=stride, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.encoder_cnn_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.encoder_pos = SoftPositionEmbed(32, (h // stride, w // stride))
        self.layer_norm = nn.LayerNorm([32, h//stride, w//stride])  # The original seems to normalise accross channels!
        self.mlp_0 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.mlp_1 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        res2 = self.encoder_cnn_0(x)
        logger.debug_once(f"res2 {res2.shape}")
        res3 = self.encoder_cnn_1(res2)
        logger.debug_once(f"res3 {res3.shape}")
        res4 = self.mlp_0(self.layer_norm(self.encoder_pos(res3)))
        logger.debug_once(f"res4 {res4.shape}")
        res5 = self.mlp_1(res4)
        logger.debug_once(f"res5 {res5.shape}")
        return {
            'res2': res2,
            'res3': res3,
            'res4': res4,
            'res5': res5,
        }


@BACKBONE_REGISTRY.register()
class SACNN_SMALL(SlotAttentionCNNBackboneSMALL, Backbone):
    def __init__(self, cfg, input_shape):
        pretrain_img_size = cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE
        patch_size = cfg.MODEL.SWIN.PATCH_SIZE
        in_chans = 3
        embed_dim = cfg.MODEL.SWIN.EMBED_DIM
        depths = cfg.MODEL.SWIN.DEPTHS
        num_heads = cfg.MODEL.SWIN.NUM_HEADS
        window_size = cfg.MODEL.SWIN.WINDOW_SIZE
        mlp_ratio = cfg.MODEL.SWIN.MLP_RATIO
        qkv_bias = cfg.MODEL.SWIN.QKV_BIAS
        qk_scale = cfg.MODEL.SWIN.QK_SCALE
        drop_rate = cfg.MODEL.SWIN.DROP_RATE
        attn_drop_rate = cfg.MODEL.SWIN.ATTN_DROP_RATE
        drop_path_rate = cfg.MODEL.SWIN.DROP_PATH_RATE
        norm_layer = nn.LayerNorm
        ape = cfg.MODEL.SWIN.APE
        patch_norm = cfg.MODEL.SWIN.PATCH_NORM
        use_checkpoint = cfg.MODEL.SWIN.USE_CHECKPOINT
        frozen_stages = cfg.MODEL.BACKBONE.FREEZE_AT

        ic = input_shape.channels
        h, w = cfg.UNSUPVIDSEG.RESOLUTION[-2:]
        shape = (ic, h, w)
        stride = 2
        super().__init__(input_shape=shape, stride=stride)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 1,
            "res3": 1,
            "res4": 1,
            "res5": 1,
        }
        self._out_feature_channels = {
            "res2": 32,
            "res3": 32,
            "res4": 64,
            "res5": 64,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        return SlotAttentionCNNBackboneSMALL.forward(self, x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    # Copy some signatures

    @property
    def size_divisibility(self):
        return 0

    def _freeze_stages(self):
        logger.warning(f"_freeze_stages() called for SMALL CNN backbone -- This is NOOP")

    def init_weights(self, pretrained=None):
        logger.warning(f"init_weights() called for SMALL CNN backbone -- This is NOOP")

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        self.encoder_cnn_0.train(mode)
        self.encoder_cnn_1.train(mode)
        self.encoder_pos.train(mode)
        self.layer_norm.train(mode)
        self.mlp_0.train(mode)
        self.mlp_1.train(mode)
        self._freeze_stages()
