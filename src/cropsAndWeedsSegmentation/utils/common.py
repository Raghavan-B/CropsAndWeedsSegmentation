import os
import yaml
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.constants import DEVICE
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
from ensure import ensure_annotations
from pathlib import Path
from typing import Any,Union,Type
from box.exceptions import BoxValueError
from box import ConfigBox
import sys
import json
import torch
import albumentations as A


def get_train_augs():
    return A.Compose([
        A.HorizontalFlip(0.5),
        A.VerticalFlip(0.43),
        A.RandomBrightnessContrast(0.2),
        A.GaussianBlur(p=0.1),
        A.GridDistortion(p=0.2),
        A.RandomRotate90(p=0.5),
    ])

@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    '''
    Description:

    Args:

    Returns:
    
    '''
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'Yaml file: {path_to_yaml} loaded successfully')
            return ConfigBox(content)
    except BoxValueError as b:
        logger.error(f'Error occured {b}')
        raise ValueError('Yaml File is Empty')
    except Exception as e:
        logger.error(f'Error occured {e}')
        raise SegmentationException(e,sys)
    
@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    '''
    Description:

    Args:

    Returns:
    '''
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f'Created directory at: {path}')


def save_json(path:Path, data:dict) -> None:
    '''
    Description:

    Args:

    Returns:
    '''
    with open(path,'w') as f:
        json.dump(data,f,indent=4)

    logger.info(f'Json file saved at {path}')


@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    '''
    Description:

    Args:

    Returns:
    '''
    with open(path,'w') as f:
        content = json.load(f)

    logger.info(f'Json file loaded from  {path}')
    return ConfigBox(content)


def save_model(path:Union[str, Path],model) -> None:
    '''
    Description:

    Args:

    Returns:
    '''
    torch.save(model.state_dict(),path)
    logger.info(f'Saved the model successfully at path {path}')

@ensure_annotations
def load_model(path:Path, model:torch.nn.Module) -> torch.nn.Module:
    '''
    Description:

    Args:

    Returns:
    '''
    model.load_state_dict(torch.load(path,map_location=DEVICE))
    logger.info(f'The model is successfully loaded from path {path}')
    return model
    
