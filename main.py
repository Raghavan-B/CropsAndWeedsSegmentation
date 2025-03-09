from src.cropsAndWeedsSegmentation.logging.logger import logger
from src.cropsAndWeedsSegmentation.exception.exception import SegmentationException
import sys

if __name__ == '__main__':
    try:
        logger.info('Project implementation has been started!!!')
    except Exception as e:
        logger.error(f'Error occured: {e}')
        raise SegmentationException(e,sys)
        