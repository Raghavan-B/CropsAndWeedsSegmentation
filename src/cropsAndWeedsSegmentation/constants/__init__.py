from pathlib import Path

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path('params.yaml')
SCHEMA_FILE_PATH = Path('schema.yaml')

COLOR_TO_LABEL = {
    (0,0,0):0, ##background
    (255,0,0):1, ## Weed
    (0,255,0):2 ## Crop
}

LABEL_TO_COLOR = {
    0: (0, 0, 0),       # Background - Black
    1: (255, 0, 0),     # Weed - Red
    2: (0, 255, 0)      # Crop - Green
}



