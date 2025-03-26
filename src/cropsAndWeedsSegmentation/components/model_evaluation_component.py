from src.cropsAndWeedsSegmentation.entity.config_entity import ModelEvaluationConfig
from src.cropsAndWeedsSegmentation.utils.model_train_utils import eval_fn
from src.cropsAndWeedsSegmentation.utils.model_eval_utils import inference_speed
from src.cropsAndWeedsSegmentation.utils.common import save_json
from src.cropsAndWeedsSegmentation.utils.model_class_utils import SegmentationModel
from src.cropsAndWeedsSegmentation.constants import DEVICE
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

torch.serialization.add_safe_globals([SegmentationModel])

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config

    def create_model_architecture(self)->torch.nn.Module:
        '''
        
        '''
        model_arch = smp.Segformer(
            encoder_name=self.config.enoder,
            encoder_weights=self.config.weights,
            in_channels=self.config.in_channels,
            classes=self.config.classes,
            activation=None
        )
        return model_arch
    
    def create_model(self,model_arch:torch.nn.Module)-> torch.nn.Module:
        '''

        '''
        return SegmentationModel(arc=model_arch).to(DEVICE)
    
    def get_model_with_weights(self):
        model_path = Path(self.config.model_path)
        # model = load_model(model_path,model_arc)
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