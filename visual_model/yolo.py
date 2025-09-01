import os

from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DEFAULT_INITIAL_CONFIG = 'yolo11n-obb.yaml'
DEFAULT_CONFIG = '../models/yolo11/yolo11n-obb-dota.yaml'
DEFAULT_MODEL_PATH = '../models/yolo11/yolo11n-obb.pt'
LATEST_MODEL_PATH = '../models/yolo11/yolo11n-obb-latest.pt'


class Yolo11:
    def __init__(
            self,
            model_name_or_path: str = None,
            config: str = DEFAULT_INITIAL_CONFIG,
            init: bool = False
    ):
        if init:
            config = DEFAULT_INITIAL_CONFIG
            model_name_or_path = DEFAULT_MODEL_PATH
            self.model = YOLO(config).load(model_name_or_path)
        else:
            if not model_name_or_path:
                model_name_or_path = LATEST_MODEL_PATH
            self.model = YOLO(model_name_or_path)
        print(f'load model from {model_name_or_path}, config: {config}')

    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def show(self):
        print(self.model)


if __name__ == '__main__':
    Yolo11().show()
