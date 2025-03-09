from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass
class DataIngestionConfig:
    root_dir: Path

    train_img_dir: Path
    train_mask_dir: Path
    
    test_img_dir: Path
    test_mask_dir: Path
    
    val_img_dir: Path
    val_mask_dir: Path
    
    others_img_dir: Path
    
    mongo_uri: str
    database_name: str
    collection_name: str

@dataclass
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    data_dir: Path
    batch_size: int
    shuffle:bool



