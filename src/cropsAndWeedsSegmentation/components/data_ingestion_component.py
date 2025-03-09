from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.entity.config_entity import (DataIngestionConfig)

from PIL import Image
import io
from typing import Tuple
import pymongo
import os

## component
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def get_image_docs_from_mongo_db(self)->Tuple[pymongo.cursor.Cursor, pymongo.cursor.Cursor, pymongo.cursor.Cursor, pymongo.cursor.Cursor]:
        '''
        '''
        client = MongoClient(self.config.mongo_uri, tlsCAFile= certifi.where(),server_api = ServerApi('1'))
        db = client[self.config.database_name]
        collection = db[self.config.collection_name]

        train_img_docs = collection.find({'category':'train'})
        val_img_docs = collection.find({'category':'val'})
        test_img_docs = collection.find({'category':'test'})
        others_img_docs = collection.find({'category':'others'})
        
        logger.info('All the required documents retrieved successfully!!')
        return train_img_docs,val_img_docs,test_img_docs,others_img_docs


    def store_img_docs_locally(self,docs:pymongo.cursor.Cursor,split:str)->None:
        '''
        '''
        for doc in docs:

            img = Image.open(io.BytesIO(doc["image"]))
            img_name = f'{doc["filename"]}.jpg'
            img_filepath = os.path.join(self.config.root_dir,doc["category"],'img',img_name)
            img.save(img_filepath)

            if doc["mask"]!=None:    
                mask = Image.open(io.BytesIO(doc["mask"]))
                mask_name = f'{doc["filename"]}.png'
                mask_filepath = os.path.join(self.config.root_dir,doc["category"],'mask',mask_name)
                mask.save(mask_filepath)
            
        logger.info(f'All the images and respective masks of {split} set were loaded to {self.config.root_dir} successfully')


## root dir --> root_dir/category/