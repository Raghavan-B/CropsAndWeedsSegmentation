from src.cropsAndWeedsSegmentation.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.cropsAndWeedsSegmentation.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.cropsAndWeedsSegmentation.pipeline.prediction_pipeline import PredictionPipeline

from src.cropsAndWeedsSegmentation.utils.common import save_json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException


import sys
from dotenv import load_dotenv
import os

load_dotenv()
mongo_uri = os.getenv('MONGO_URL')
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
    obj = DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion(mongo_uri)
    logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)

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

STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
    obj = ModelTrainerTrainingPipeline()
    avg_epoch_train_loss,avg_epoch_train_pxl_acc,avg_epoch_valid_loss,avg_epoch_valid_pxl_acc = obj.initiate_model_training(trainloader,validloader)
    metrics = {
    'Average train loss per epoch':avg_epoch_train_loss,
    'Average validation loss per epoch':avg_epoch_valid_loss,
    'Average train pixel accuracy per epoch':avg_epoch_train_pxl_acc,
    'Average validation pixel accuracy per epoch':avg_epoch_valid_pxl_acc
    }
    metrics_filepath = os.path.join('artifacts/model_trainer','training_metrics.json')
    save_json(metrics_filepath,metrics)
    logger.info('Training Metrics has been saved successfully in artifacts/model_trainer')
    logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')

except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)

STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
    obj = ModelEvaluationPipeline()
    obj.initiate_model_evaluation(testloader)
    logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)


STAGE_NAME = "Prediction Pipeline Stage"
try:
    logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
    img_path = "artifacts/data_ingestion/others/img/20918823_frame_002839.jpg"
    obj = PredictionPipeline(model_weights_path=Path('artifacts/model_trainer/model_weights.pth'))
    pred_mask = obj.segment_images(img_path)
    logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
except Exception as e:
    logger.error(f'Error occured : {e}')
    raise SegmentationException(e,sys)

