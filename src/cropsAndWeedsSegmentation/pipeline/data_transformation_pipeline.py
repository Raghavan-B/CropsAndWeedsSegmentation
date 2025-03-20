from src.cropsAndWeedsSegmentation.utils.common import get_train_augs
from src.cropsAndWeedsSegmentation.utils.data_transformation_utils import convert_color_mask_to_label
from src.cropsAndWeedsSegmentation.config.configuration import ConfigurationManager
from src.cropsAndWeedsSegmentation.components.data_transformation_component import DataTransformation
from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
import sys
import matplotlib.pyplot as plt

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        trainset = data_transformation.create_dataset_class(split='train',
                                                            convert_color_mask_to_label=convert_color_mask_to_label,
                                                            augmentations = get_train_augs())
        logger.info(f'Size of train dataset {len(trainset)}')
        logger.info('Train Dataset has been created successfully!!')
        image,mask = trainset[0]
        # plt.subplot(1,2,1)
        # plt.imshow(image.permute(1,2,0))
        # plt.subplot(1,2,2)
        # plt.imshow(mask.permute(1,2,0))
        # plt.show()
        testset = data_transformation.create_dataset_class(split='test',
                                                            convert_color_mask_to_label=convert_color_mask_to_label,
                                                            )
        logger.info(f'Size of the test dataset : {len(testset)}')
        logger.info('Test Dataset has been created successfully!!')
        
        validset = data_transformation.create_dataset_class(split='val',
                                                            convert_color_mask_to_label=convert_color_mask_to_label)
        logger.info(f'Size of the validation dataset {len(validset)}')
        logger.info('Validation Dataset has been created successfully!!')

        logger.info('Dataloader creation has been started!')
        trainloader = data_transformation.create_dataloader(trainset,shuffle=True)
        testloader = data_transformation.create_dataloader(testset)
        validloader = data_transformation.create_dataloader(validset)
        logger.info('All Dataloaders has been created successfully!!')

        return trainloader,testloader,validloader
    

STAGE_NAME = "Data Transformation Stage"

if __name__ == '__main__':
    try:
        logger.info(f'>>>> Stage: {STAGE_NAME} started <<<<')
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f'>>>> Stage: {STAGE_NAME} Completed <<<<\n\nx=====x')
    except Exception as e:
        logger.error(f'Error occured : {e}')
        raise SegmentationException(e,sys)


