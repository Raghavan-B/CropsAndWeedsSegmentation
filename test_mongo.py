from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
from box import ConfigBox

load_dotenv()

MONGO_URI = os.getenv('MONGO_URL')
DATABASE = 'images_db'
COLLECTION = 'images_dataset_collection'
client = MongoClient(MONGO_URI, tlsCAFile= certifi.where(),server_api = ServerApi('1'))
db = client[DATABASE]
collection = db[COLLECTION]


try:
    image_doc = ConfigBox(collection.find({'category':'val'}))
    print(image_doc.filename)
    # img = Image.open(io.BytesIO(image_doc['image']))
    # img.save(f'{image_doc['filename']}.jpg')

    # if image_doc['mask']:
    #     mask = np.array(Image.open(io.BytesIO(image_doc['mask'])))
    #     print(np.unique(mask))
        # mask.save(f'{image_doc['filename']}.jpg')

    # img_docs = collection.count_documents({'category':'train'})
    # print(img_docs)
    
   
except Exception as e:
    print(e)