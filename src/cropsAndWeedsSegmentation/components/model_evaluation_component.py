from src.cropsAndWeedsSegmentation.entity.config_entity import ModelEvaluationConfig
from src.cropsAndWeedsSegmentation.utils.model_train_utils import eval_fn
from src.cropsAndWeedsSegmentation.utils.model_eval_utils import inference_speed
from src.cropsAndWeedsSegmentation.utils.common import load_model,save_json
from src.cropsAndWeedsSegmentation.utils.model_class_utils import SegmentationModel

from pathlib import Path
import torch
import segmentation_models_pytorch as smp


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
    
    def create_model(self,model_arch:torch.nn.Module,device:torch.device)-> torch.nn.Module:
        '''

        '''
        return SegmentationModel(arc=model_arch,
                                 device= device).to(device)
    
    def get_model_with_weights(self,model_arc,device:str = 'cuda:0'):
        model_path = Path(self.config.model_path)
        model = load_model(model_path,model_arc,device)
        return model
    
    def evaluate_model(self,testloader,model,DEVICE):
        ## pixel accuracy and test_loss 
        avg_test_loss,avg_pixel_acc = eval_fn(testloader,model,DEVICE)
        test_img,_ = next(iter(testloader))
        test_img = test_img[0].unsqueeze(0).to(DEVICE)
        test_inference_speed = inference_speed(image = test_img,model=model)
        return avg_test_loss,avg_pixel_acc,test_inference_speed
    
    def import_metrics_to_json(self,metrics:dict, path:Path):
        save_json(path,metrics)