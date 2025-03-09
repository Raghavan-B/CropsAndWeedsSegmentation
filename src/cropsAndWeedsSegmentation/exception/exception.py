import sys
from src.cropsAndWeedsSegmentation.logging import logger

class SegmentationException(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f'\nError occured in Python Script Name: {self.file_name} \nLine Number: {self.lineno} \nError Message: {str(self.error_message)}'
    