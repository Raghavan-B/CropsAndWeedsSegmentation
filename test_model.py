import mlflow.pytorch
import mlflow
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from src.cropsAndWeedsSegmentation.constants import DEVICE,LABEL_TO_COLOR
from src.cropsAndWeedsSegmentation.utils.data_transformation_utlis import colorize_label_mask
from src.cropsAndWeedsSegmentation.pipeline.prediction_pipeline import PredictionPipeline

mlflow.set_tracking_uri('https://dagshub.com/Raghavan-B/CropsAndWeedsSegmentation.mlflow')
model_name = "models:/Best_model_v2/latest"
model_path = "cached_model"
os.makedirs(model_path,exist_ok=True)

try:
    image_path = "artifacts/data_ingestion/others/img/20824007_frame_000075.jpg"
    obj = PredictionPipeline(None,model_name=model_name,model_path=model_path)
    mask = obj.predict(image_path)
    plt.imshow(mask)
    plt.show()
except Exception as e:
    print(e)
