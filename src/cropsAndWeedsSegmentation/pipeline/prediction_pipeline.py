import numpy as np
from PIL import Image
from src.cropsAndWeedsSegmentation.utils.common import load_model
from pathlib import Path
import torchvision.transforms as transforms
import torch
from src.cropsAndWeedsSegmentation.utils.model_class_utils import SegmentationModel
import segmentation_models_pytorch as smp
from src.cropsAndWeedsSegmentation.utils.data_transformation_utlis import colorize_label_mask
from src.cropsAndWeedsSegmentation.constants import LABEL_TO_COLOR,DEVICE

import matplotlib.pyplot as plt
torch.serialization.add_safe_globals([SegmentationModel])
class PredictionPipeline:
    def __init__(self,model_weights_path):
        self.model_weights_path = model_weights_path

    def initiate_model_arch(self):
        '''
        
        '''
        model_arch = smp.Segformer(
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None
        )
        return model_arch
    
    def create_model(self,model_arch:torch.nn.Module)-> torch.nn.Module:
        '''

        '''
        return SegmentationModel(arc=model_arch).to(DEVICE)
    
    def initiate_model(self):
        model_arch = self.initiate_model_arch()
        
        # model = load_model(self.model_weights_path,model_arch)
        model = torch.load(self.model_weights_path,map_location=DEVICE,weights_only=False)
        return model
    
    def segment_images(self,image,mask_image=None):
        model = self.initiate_model()
        image = Image.open(image)
        if image.size != (224,224):
            image = image.resize((224,224),Image.LANCZOS)
        image = np.array(image)
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        image = torch.tensor(image)/255.0
        model.eval()
        with torch.no_grad():
            pred_logits = model(image.unsqueeze(0).to(DEVICE))
            pred_mask = pred_logits.argmax(dim=1)
        image = image.permute(1,2,0)
        pred_mask = pred_mask.cpu().numpy().squeeze(0)
        colored_mask = colorize_label_mask(pred_mask,LABEL_TO_COLOR)
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(colored_mask)
        plt.show()