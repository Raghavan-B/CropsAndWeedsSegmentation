from src.cropsAndWeedsSegmentation.entity.config_entity import DataTransformationConfig
from src.cropsAndWeedsSegmentation.constants import COLOR_TO_LABEL,LABEL_TO_COLOR
from src.cropsAndWeedsSegmentation.utils.data_transformation_utlis import colorize_label_mask,convert_color_mask_to_label
from src.cropsAndWeedsSegmentation.utils.dataset_class_utils import CropsAndWeedsDataset
from typing import Callable
from ensure import ensure_annotations
from torch.utils.data import Dataset,DataLoader

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config = config

    @ensure_annotations
    def create_dataset_class(self,split:str, 
                             convert_color_mask_to_label:Callable, 
                             augmentations:Callable=None )->Dataset:
        '''
        '''
        dataset = CropsAndWeedsDataset(dataset=self.config.data_dir, 
                                       split=split, 
                                       augmentations=augmentations, 
                                       convert_color_mask_to_label=convert_color_mask_to_label)
        return dataset
    
    @ensure_annotations
    def create_dataloader(self,dataset:Dataset,shuffle:bool = False)->DataLoader:
        '''
        '''
        if shuffle:
            self.config.shuffle=True
        return DataLoader(dataset, 
                          self.config.batch_size, 
                          shuffle=self.config.shuffle)
        