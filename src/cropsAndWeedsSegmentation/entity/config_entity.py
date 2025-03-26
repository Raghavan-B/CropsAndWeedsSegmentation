from dataclasses import dataclass
from pathlib import Path

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

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    epochs: int
    lr: float
    weight_decay: float
    enoder: str
    weights: str
    architecture: str
    in_channels: int
    classes: int
    batch_size: int

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    enoder: str
    weights: str
    in_channels: int
    classes: int





