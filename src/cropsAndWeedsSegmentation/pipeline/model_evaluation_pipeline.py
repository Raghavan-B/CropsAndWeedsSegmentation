from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.model_evaluation_component import ModelEvaluation
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.constants import DEVICE

from pathlib import Path
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self,testloader):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_eval_config()

        model_evaluation = ModelEvaluation(model_evaluation_config)
        
        model_arch = model_evaluation.create_model_architecture()
        logger.info("Model architecture created successfully!!")
        model = model_evaluation.create_model(model_arch,DEVICE)
        logger.info("Model has been created successfully")
        model = model_evaluation.get_model_with_weights(model)
        logger.info(f"Model weights are loaded successfully from {model_evaluation_config.model_path}")
        avg_test_loss,avg_pixel_acc,test_inference_speed = model_evaluation.evaluate_model(testloader,model,DEVICE)
        metrics_file_path = Path(model_evaluation_config.root_dir)/"testing_metrics.json"
        metrics = {
            "avg_test_loss":avg_test_loss,
            "avg_pixel_acc": avg_pixel_acc,
            "inference_speed (50 images) (ms)":test_inference_speed*1000
        }
        model_evaluation.import_metrics_to_json(metrics,metrics_file_path)
        logger.info("Metrics were saved!!")