import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
try:
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
except Exception:
    DeepLabV3_ResNet101_Weights = None


class DeepLabV3Plus(nn.Module):
    """Wrapper for DeepLabV3+ style segmentation using ResNet101 backbone.

    - Uses torchvision's DeepLabV3 ResNet101 as base.
    - Exposes `.backbone` and `.head` to align with existing training code.
    - Provides `lock_backbone()` and accepts `comp_drop` in forward for compatibility.
    """

    def __init__(self, nclass: int, pretrained: bool = True):
        super().__init__()
        # Use torchvision's implementation; for offline environments avoid weight download by setting pretrained=False.
        # Note: torchvision provides DeepLabV3 (not strictly Plus). This wrapper aligns interfaces and allows training.
        if DeepLabV3_ResNet101_Weights is not None and pretrained:
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        else:
            weights = None
        self.model = deeplabv3_resnet101(weights=weights)
        # Replace classifier head to match number of classes
        # ResNet101 backbone output channels = 2048
        self.model.classifier = DeepLabHead(2048, nclass)

        # Expose backbone and head for optimizer grouping and logging
        self.backbone = self.model.backbone
        self.head = self.model.classifier

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, comp_drop: bool = False):
        # comp_drop is ignored; kept for API compatibility
        out = self.model(x)
        return out["out"]