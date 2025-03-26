from src.cropsAndWeedsSegmentation.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH,SCHEMA_FILE_PATH
from src.cropsAndWeedsSegmentation.utils.common import read_yaml,create_directories
from src.cropsAndWeedsSegmentation.entity.config_entity import (DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath = PARAMS_FILE_PATH,schema_file_path = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self,mongo_uri:str)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir,config.train_img_dir,config.train_mask_dir,config.test_img_dir,config.test_mask_dir,config.val_img_dir,config.val_mask_dir,config.others_img_dir])  

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            
            train_img_dir=config.train_img_dir,
            train_mask_dir= config.train_mask_dir,

            test_img_dir=config.test_img_dir,
            test_mask_dir= config.test_mask_dir,

            val_img_dir=config.val_img_dir,
            val_mask_dir= config.val_mask_dir,

            others_img_dir=config.others_img_dir,

            mongo_uri=mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name
        )      
        return data_ingestion_config
    
    def get_data_validation_config(self)->DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.InputImg

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema
        )

        return data_validation_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            shuffle=config.shuffle
        )

        return data_transformation_config
    
    def get_model_training_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.Segformer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_name=config.model_name,
            epochs=params.epochs,
            lr = params.lr,
            weight_decay=params.weight_decay,
            enoder= params.encoder,
            weights=params.weights,
            architecture=params.architecture,
            in_channels=params.in_channels,
            classes=params.classes,
            batch_size=16
        )

        return model_trainer_config
    
    def get_model_eval_config(self)->ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.Segformer

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path= config.model_path,
            enoder = params.encoder,
            weights = params.weights,
            in_channels = params.in_channels,
            classes = params.classes
        )
    


    
    

    
    

    

