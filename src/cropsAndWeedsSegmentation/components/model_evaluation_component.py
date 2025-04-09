from src.cropsAndWeedsSegmentation.entity.config_entity import ModelEvaluationConfig
from src.cropsAndWeedsSegmentation.utils.model_train_utils import eval_fn
from src.cropsAndWeedsSegmentation.utils.model_eval_utils import inference_speed
from src.cropsAndWeedsSegmentation.utils.common import save_json
from src.cropsAndWeedsSegmentation.constants import DEVICE
from pathlib import Path
import torch


class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config
    
    def get_model_with_weights(self):
        model_path = Path(self.config.model_path)
        model = torch.load(model_path,map_location=DEVICE,weights_only=False)
        return model
    
    def evaluate_model(self,testloader,model):
        ## pixel accuracy and test_loss 
        avg_test_loss,avg_pixel_acc = eval_fn(testloader,model)
        test_img,_ = next(iter(testloader))
        test_img = test_img[0].unsqueeze(0).to(DEVICE)
        test_inference_speed = inference_speed(image = test_img,model=model)
        return avg_test_loss,avg_pixel_acc,test_inference_speed
    
    def import_metrics_to_json(self,metrics:dict, path:Path):
        save_json(path,metrics)