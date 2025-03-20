import numpy as np
from PIL import Image
import torch
import os
import mlflow
import mlflow.pytorch

from src.cropsAndWeedsSegmentation.utils.data_transformation_utils import colorize_label_mask
from src.cropsAndWeedsSegmentation.constants import LABEL_TO_COLOR


class PredictionPipeline:
    def __init__(self,model_name,model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_file = os.path.join(self.model_path,"data/model.pth")

    def save_model_from_mlflow(self):
        if not os.path.exists(self.model_file):
            model = mlflow.pytorch.load_model(self.model_name, map_location = torch.device('cpu'))
            mlflow.pytorch.save_model(model, self.model_path)
            print("Model is saved")
        else:
            print('model already exists')
    
    def load_model_from_local(self):
        model = torch.load(self.model_file,map_location=torch.device('cpu'), weights_only=False)
        return model
    
    def predict(self,img_path):
        self.save_model_from_mlflow()
        model = self.load_model_from_local()
        img = Image.open(img_path)
        if img.size != (224,224):
            img = img.resize((224,224),Image.LANCZOS)
        img = np.array(img)
        img = np.transpose(img,(2,0,1)).astype(np.float32)
        img = torch.tensor(img)/255.0

        model.eval()
        with torch.no_grad():
            pred_logits = model(img.unsqueeze(0).to(torch.device("cpu")))
            pred_mask = pred_logits.argmax(dim = 1)
        img = img.permute(1,2,0)
        pred_mask = pred_mask.cpu().numpy().squeeze(0)
        colored_mask = colorize_label_mask(pred_mask,LABEL_TO_COLOR)
        return colored_mask
