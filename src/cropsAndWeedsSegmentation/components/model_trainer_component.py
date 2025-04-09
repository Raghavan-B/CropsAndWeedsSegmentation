from src.cropsAndWeedsSegmentation.entity.config_entity import (ModelTrainerConfig)
from src.cropsAndWeedsSegmentation.utils.model_class_utils import AgriFormer
from src.cropsAndWeedsSegmentation.constants import DEVICE

import torch
import segmentation_models_pytorch as smp

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    def create_segmentation_model(self)-> torch.nn.Module:
        return AgriFormer(encoder_name=self.config.enoder,
                          num_classes=self.config.classes,
                          in_channels=self.config.in_channels).to(DEVICE)


    
