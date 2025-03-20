import os
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from src.cropsAndWeedsSegmentation.entity.config_entity import (DataValidationConfig)

class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config = config
        self.type = {
            "JpegImageFile": JpegImageFile
        }

    def validate_dataset(self)->bool:
        folders = ['train','test','val','others']
        for folder in folders:
            folder_path = os.path.join(self.config.data_dir,folder,'img')
            for img_file in os.listdir(folder_path):
                img_filePath = os.path.join(folder_path,img_file)
                img = Image.open(img_filePath)

                if not isinstance(img,self.type[self.config.all_schema.type]) and img.size != (self.config.all_schema.height,self.config.all_schema.width):
                    with open(self.config.STATUS_FILE,'w') as file:
                        file.write(f'Validation status: False')
                    return False
                
        with open(self.config.STATUS_FILE,'w') as file:
            file.write(f'Validation status: True')
        return True

