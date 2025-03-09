from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.data_validation_component import DataValidation
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
from src.cropsAndWeedsSegmentation.logging.logger import logger
import sys

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_validation(self)->bool:
            config = ConfigurationManager()
            data_validaion_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validaion_config)
            validation_status = data_validation.validate_dataset()
            return validation_status
    
STAGE_NAME = "Data Validation Stage"

if __name__ == '__main__':
    try:
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataValidationTrainingPipeline()
        validation_status = obj.initiate_data_validation()
        logger.info(f'Status: {validation_status}')
        logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
    except Exception as e:
        logger.error(f'Error occured : {e}')
        raise SegmentationException(e,sys)

