from dotenv import load_dotenv
from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.data_ingestion_component import DataIngestion
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
import sys
import os
load_dotenv()


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self,mongo_uri:str)->None:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config(mongo_uri=mongo_uri)
        
        data_ingestion = DataIngestion(data_ingestion_config)
        
        train_img_docs,val_img_docs,test_img_docs,others_img_docs = data_ingestion.get_image_docs_from_mongo_db()
        ## storing train docs
        data_ingestion.store_img_docs_locally(train_img_docs,'train')
        ## storing validation docs
        data_ingestion.store_img_docs_locally(val_img_docs,'val')
        ## storing test docs
        data_ingestion.store_img_docs_locally(test_img_docs,'test')
        ## storing others docs
        data_ingestion.store_img_docs_locally(others_img_docs,'others')


mongo_uri = os.getenv('MONGO_URL')
STAGE_NAME = "Data Ingestion Stage"


if __name__ == '__main__':
    try:
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion(mongo_uri)
        logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
    except Exception as e:
        logger.error(f'Error occured : {e}')
        raise SegmentationException(e,sys)
