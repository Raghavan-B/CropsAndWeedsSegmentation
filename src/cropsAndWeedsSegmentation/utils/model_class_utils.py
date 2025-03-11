import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torch
import torch.nn.functional as F
from src.cropsAndWeedsSegmentation.constants import DEVICE

class SegmentationModel(nn.Module):
    def __init__(self,arc):
        super(SegmentationModel,self).__init__()
        self.arc = arc

    def forward(self,images,masks = None):
        # masks = masks.to(self.device)
        logits = self.arc(images)
        preds = F.softmax(logits,dim = 1)
        if masks!=None:
            loss1 = DiceLoss(mode= 'multiclass',from_logits=True,ignore_index=0)(logits,masks)
            loss2 = nn.CrossEntropyLoss(weight = torch.tensor([0.1, 1.0, 1.0]).to(DEVICE))(logits,masks.long())
            return preds,loss1+loss2
        return preds
