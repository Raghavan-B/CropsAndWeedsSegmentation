from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.model_evaluation_component import ModelEvaluation
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.constants import DEVICE
from src.cropsAndWeedsSegmentation.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException

import torch
import sys
from pathlib import Path
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self,testloader):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_eval_config()

        model_evaluation = ModelEvaluation(model_evaluation_config)
        
        model = model_evaluation.get_model_with_weights()
        logger.info(f"Model is loaded successfully from {model_evaluation_config.model_path}")
        avg_test_loss,avg_pixel_acc,test_inference_speed = model_evaluation.evaluate_model(testloader,model)
        metrics_file_path = Path(model_evaluation_config.root_dir)/"testing_metrics.json"
        metrics = {
            "avg_test_loss":avg_test_loss,
            "avg_pixel_acc": avg_pixel_acc,
            "inference_speed (50 images) (ms)":test_inference_speed*1000
        }
        model_evaluation.import_metrics_to_json(metrics,metrics_file_path)
        logger.info("Metrics were saved!!")

if __name__ == '__main__':
    STAGE_NAME = "Model Evaluation Stage"
    try:
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataTransformationTrainingPipeline()
        trainloader,testloader,validloader = obj.initiate_data_transformation()


        obj = ModelEvaluationPipeline()
        obj.initiate_model_evaluation(testloader)
        logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
    except Exception as e:
        logger.error(f'Error occured : {e}')
        raise SegmentationException(e,sys)