import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torchvision.models.resnet import resnet18, resnet50, model_urls
from torchvision.models._utils import IntermediateLayerGetter

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from utils.log import getLogger

logger = getLogger(__name__)

class ResNet(nn.Module):
    nets = {
        'resnet18': resnet18,
        'resnet50': resnet50,
    }

    features = {
        'resnet18': {
            "res2": 64,
            "res3": 128,
            "res4": 256,
            "res5": 512,
        },
        'resnet50': {
            "res2": 256,
            "res3": 512,
            "res4": 1024,
            "res5": 2048,
        }
    }

    def __init__(self, model='ResNet18', pretrained=False):
        super(ResNet, self).__init__()
        resnet = self.nets[model.lower()](pretrained=pretrained, progress=True)
        if pretrained:
            url = model_urls[model.lower()]
            logger.info(f"Loading pretrained {model} from {url}")
            state_dict = load_state_dict_from_url(url,
                                                  progress=True)
            for k, v in state_dict.items():
                logger.debug(f"Loading values for {k} {v.shape}")
            resnet.load_state_dict(state_dict)
        else:
            logger.info(f"Model is not pretrained")
        self.resnet = IntermediateLayerGetter(resnet, {
            'layer1': 'res2',
            'layer2': 'res3',
            'layer3': 'res4',
            'layer4': 'res5'
        })

    def forward(self, x):
        return self.resnet(x)

@BACKBONE_REGISTRY.register()
class ResNetTV(ResNet, Backbone):
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

        model_str = "ResNet" + str(cfg.MODEL.RESNETS.DEPTH)
        use_pretraining = cfg.MODEL.WEIGHTS is not None
        if use_pretraining:
            logger.critical(f"Using pretrained weights for {model_str}")
        super().__init__(model=model_str, pretrained=use_pretraining)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = self.features[model_str.lower()]


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """

        return self.resnet(x)

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
        logger.warning(f"_freeze_stages() called for ResNetTV -- This is NOOP")

    def init_weights(self, pretrained=None):
        logger.warning(f"init_weights() called for ResNetTV -- This is NOOP")

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        self.resnet.train(mode)
        self._freeze_stages()
