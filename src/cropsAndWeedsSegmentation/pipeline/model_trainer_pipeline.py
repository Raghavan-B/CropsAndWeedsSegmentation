from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.model_trainer_component import ModelTrainer
from src.cropsAndWeedsSegmentation.utils.model_train_utils import train_fn,eval_fn
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
from src.cropsAndWeedsSegmentation.utils.common import save_json
from src.cropsAndWeedsSegmentation.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline

import torch.optim as optim
import mlflow
import mlflow.pytorch
import dagshub
import torch

from tqdm import tqdm
import os
import sys




class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self,trainloader,validloader):

        dagshub.init(repo_owner='Raghavan-B', repo_name='CropsAndWeedsSegmentation', mlflow=True)
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_params = {
                'Model architecture': 'Proposed model',
                'Encoder': model_trainer_config.enoder,
                'Weights':model_trainer_config.weights,
                'in_channels':model_trainer_config.in_channels,
                'classes':model_trainer_config.classes,
                'learning_rate': model_trainer_config.lr,
                'batch_size':model_trainer_config.batch_size,
                'epochs':model_trainer_config.epochs,
                'weight_decay': model_trainer_config.weight_decay

        }
        model_trainer = ModelTrainer(model_trainer_config)
        
        model = model_trainer.create_segmentation_model()
        logger.info("Model has been created successfully")

        epochs = model_trainer_config.epochs
        lr = model_trainer_config.lr
        weight_decay = model_trainer_config.weight_decay
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
        with mlflow.start_run():
            for key,value in model_params.items():
                    mlflow.log_param(key,value)

            logger.info("Model training has been started...")
            for epoch in tqdm(range(epochs)):
                avg_epoch_train_loss,avg_epoch_train_pxl_acc = train_fn(trainloader,model,optimizer)
                
                mlflow.log_metric('train_loss',avg_epoch_train_loss,step=epoch)
                mlflow.log_metric('train_pixel_accuracy',avg_epoch_train_pxl_acc,step=epoch)

                print(f'[{model_trainer_config.architecture} | {model_trainer_config.enoder}] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_train_loss:.4f}, Pixel Accuracy: {avg_epoch_train_pxl_acc:.4f}')

                avg_epoch_valid_loss,avg_epoch_valid_pxl_acc = eval_fn(validloader,model)
                
                mlflow.log_metric('validation_loss',avg_epoch_valid_loss,step=epoch)
                mlflow.log_metric('validation_pixel_accuracy',avg_epoch_valid_pxl_acc,step=epoch)

                print(f'[{model_trainer_config.architecture} | {model_trainer_config.enoder}] Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_epoch_valid_loss:.4f}, Pixel Accuracy: {avg_epoch_valid_pxl_acc:.4f}')
                scheduler.step(avg_epoch_train_loss)

            mlflow.pytorch.log_model(model,'segmentation_model')
            logger.info("Model has been tracked successfully")
        
        model_filepath = os.path.join(model_trainer_config.root_dir,model_trainer_config.model_name)
        torch.save(model,model_filepath)

        return avg_epoch_train_loss,avg_epoch_train_pxl_acc,avg_epoch_valid_loss,avg_epoch_valid_pxl_acc

if __name__ == '__main__':
    try:
        STAGE_NAME = "Model Training Stage"
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataTransformationTrainingPipeline()
        trainloader,testloader,validloader = obj.initiate_data_transformation()

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



