from torch.utils.data import Dataset
from src.cropsAndWeedsSegmentation.constants import COLOR_TO_LABEL
import os
import numpy as np
from PIL import Image
import torch
from typing import Callable,Tuple
from pathlib import Path

class CropsAndWeedsDataset(Dataset):
    def __init__(self,dataset:Path,split:str,augmentations:Callable,convert_color_mask_to_label:Callable):
        self.dataset = dataset
        self.split = split ## not in config
        self.augmentations = augmentations
        self.convert_color_mask_to_label = convert_color_mask_to_label #not in config
        self.images_path = os.path.join(self.dataset,self.split,'img')
        self.masks_path = os.path.join(self.dataset,self.split,'mask')
        self.image_names = sorted(os.listdir(self.images_path))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,idx:int)->Tuple[torch.Tensor,torch.Tensor]:
        image_filename = self.image_names[idx]
        mask_filename = os.path.splitext(image_filename)[0]+'.png'
        image_path = os.path.join(self.images_path,image_filename)
        mask_path = os.path.join(self.masks_path,mask_filename)

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        mask = self.convert_color_mask_to_label(mask,COLOR_TO_LABEL)
        mask = np.expand_dims(mask,axis = -1)

        if self.augmentations:
            data = self.augmentations(image = image,mask = mask)
            image = data['image']
            mask = data['mask']

        image = np.transpose(image,(2,0,1)).astype(np.float32)
        mask = np.transpose(mask,(2,0,1)).astype(np.float32)

        image = torch.tensor(image)/255.0
        mask = torch.tensor(mask)
        return image,mask

 