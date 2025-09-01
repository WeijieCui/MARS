# vision_model/obb_detector.py
import torch
from ultralytics import YOLO


class OBBDetector:
    def __init__(self, weights_path, conf_thresh=0.3):
        if not weights_path:
            config = 'yolo11n-obb.yaml'
            model_name_or_path = '../models/yolo11/yolo11n-obb.pt'
            self.model = YOLO(config).load(model_name_or_path)
        else:
            model_name_or_path = '../models/yolo11/yolo11n-obb-latest.pt'
            self.model = YOLO(model_name_or_path)
        self.conf_thresh = conf_thresh

    def predict(self, img_patch):
        """Detect oriented objects in image patch"""
        results = self.model(img_patch)
        return [
            (cls, conf, poly)
            for *poly, conf, cls in results.pred[0]
            if conf >= self.conf_thresh
        ]
