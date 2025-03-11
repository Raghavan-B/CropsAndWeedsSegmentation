from src.cropsAndWeedsSegmentation.entity.config_entity import (ModelTrainerConfig)
from src.cropsAndWeedsSegmentation.utils.model_class_utils import SegmentationModel
from src.cropsAndWeedsSegmentation.constants import DEVICE

import torch
import segmentation_models_pytorch as smp

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    def create_model_architecture(self)->torch.nn.Module:
        '''
        
        '''
        model_arch = smp.Segformer(
            encoder_name=self.config.enoder,
            encoder_weights=self.config.weights,
            in_channels=self.config.in_channels,
            classes=self.config.classes,
            activation=None
        )
        return model_arch
    
    def create_model(self,model_arch:torch.nn.Module)-> torch.nn.Module:
        '''

        '''
        return SegmentationModel(arc=model_arch).to(DEVICE)


    
