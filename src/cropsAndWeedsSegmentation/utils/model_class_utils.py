import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torch
import torch.nn.functional as F
from src.cropsAndWeedsSegmentation.constants import DEVICE

from timm.layers import  LayerNorm2d
from segmentation_models_pytorch.encoders import get_encoder
    

class EffFormerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # ConvNeXt-style depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        # Swin Transformer's window attention (simplified)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
        )
        
    def forward(self, x):
        x = x + self.dwconv(x)  # Local features
        x = x + self.attn(self.norm(x))  # Global attention
        return x

class GatedCrossScaleFusion(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for ch in encoder_channels
        ])
        self.fuse = nn.Conv2d(sum(encoder_channels), 256, kernel_size=1)
        
    def forward(self, features):
        # Get the largest spatial dimensions (H, W)
        max_h = max([f.shape[2] for f in features])
        max_w = max([f.shape[3] for f in features])
        
        fused = []
        for i, (gate, f) in enumerate(zip(self.gates, features)):
            # Resize feature and gate to (max_h, max_w)
            f_resized = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            gate_resized = F.interpolate(gate(f), size=(max_h, max_w), mode='bilinear', align_corners=False)
            fused.append(f_resized * gate_resized)  # Apply gating
        
        return self.fuse(torch.cat(fused, dim=1))


class AgriFormer(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b0', num_classes=3,in_channels = 3):
        super().__init__()
        # Encoder (EfficientNet)
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights='imagenet'
        )
        # Replace last encoder block with EffFormer
        self.effformer = EffFormerBlock(dim=self.encoder.out_channels[-1])
        
        # Neck (Gated Fusion)
        self.gcf = GatedCrossScaleFusion(self.encoder.out_channels)
        
        # Head
        # self.head = DeformableHead(256, num_classes)
        self.head = nn.Sequential(
    nn.Conv2d(256, 128, kernel_size=3, padding=1),
    nn.Conv2d(128, num_classes, kernel_size=1)
)

        
    def forward(self, images, masks=None):
        # Extract multi-scale features
        features = self.encoder(images)
        # Replace last feature map with EffFormer output
        features[-1] = self.effformer(features[-1])
        # Fuse features
        x = self.gcf(features)
        # Segmentation head
        logits = self.head(x)
        preds = F.softmax(logits,dim = 1)
        if masks!=None:
            loss1 = DiceLoss(mode= 'multiclass',from_logits=True,ignore_index=0)(logits,masks)
            loss2 = nn.CrossEntropyLoss(weight = torch.tensor([0.1, 1.0, 1.0]).to(DEVICE))(logits,masks.long())
            return preds,loss1+loss2
        return preds

