from src.cropsAndWeedsSegmentation.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline

from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
import matplotlib.pyplot as plt
import torch

import sys
from dotenv import load_dotenv
import os

# load_dotenv()
# mongo_uri = os.getenv('MONGO_URL')
# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
#     obj = DataIngestionTrainingPipeline()
#     obj.initiate_data_ingestion(mongo_uri)
#     logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
# except Exception as e:
#     logger.error(f'Error occured : {e}')
#     raise SegmentationException(e,sys)

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
    obj = DataValidationTrainingPipeline()
    validation_status = obj.initiate_data_validation()
    logger.info(f'Validation Status: {validation_status}')
    logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)

STAGE_NAME = "Data Transformation Stage"
try:
    if validation_status == True:
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataTransformationTrainingPipeline()
        trainloader,testloader,validloader = obj.initiate_data_transformation()
        logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')                
        logger.info(f'Total Batches in training set: {len(trainloader)}')
        logger.info(f'Total Batches in test set: {len(testloader)}')
        logger.info(f'Total Batches in validaiton set: {len(validloader)}')
    else:
        logger.info(f'Check the data before transformation since valdiation status is {validation_status}')
except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)



    
    
