import mlflow
import matplotlib.pyplot as plt

from src.cropsAndWeedsSegmentation.pipeline.prediction_pipeline import PredictionPipeline

MODEL_PATH = "cached_model"
MODEL_NAME = "models:/Proposed_Segmentation_Model/latest"

mlflow.set_tracking_uri('https://dagshub.com/Raghavan-B/CropsAndWeedsSegmentation.mlflow')

try:
    image_path = "artifacts/data_ingestion/others/img/20824007_frame_000075.jpg"
    obj = PredictionPipeline(MODEL_NAME,MODEL_PATH)
    mask = obj.predict(image_path)
    plt.imshow(mask)
    plt.show()
    # model = torch.load('artifacts/model_trainer/model.pth',map_location=DEVICE,weights_only=False)
    # print(model)
    
except Exception as e:
    print(e)

