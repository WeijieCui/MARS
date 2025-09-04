import math
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np


# ----------------------------
# 检测器接口
# ----------------------------
class BaseDetector:
    def __init__(self):
        pass

    def infer_obb(self, crop: np.ndarray, target: int = -1) -> Tuple[
        List[Dict[str, Any]], float]:
        """
        在给定 ROI 上做检测，返回:
        - obbs: [{'cx','cy','w','h','theta','score'}, ...] (像素坐标，角度单位:度)
        - best_conf: ROI 内最优置信度（用于填表）
        默认占位实现：随机少量 OBB，概率和图像纹理简单相关
        """
        if crop.size == 0:
            return [], 0.0

        # 简易“信号”：基于边缘强度
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_score = edges.mean() / 255.0  # 0~1

        obbs = []
        best_conf = 0.0
        # 随机生成 0~3 个 OBB，概率随 edge_score 增加
        num = np.random.binomial(n=3, p=min(0.9, max(0.05, edge_score)))
        h, w = crop.shape[:2]
        for _ in range(num):
            cx = np.random.randint(w // 6, w - w // 6)
            cy = np.random.randint(h // 6, h - h // 6)
            ww = np.random.randint(max(12, w // 10), max(20, w // 3))
            hh = np.random.randint(max(12, h // 10), max(20, h // 3))
            theta = np.random.uniform(-90, 90)
            score = float(np.clip(np.random.normal(loc=0.5 + 0.4 * edge_score, scale=0.15), 0.05, 0.99))
            obbs.append(dict(cx=int(cx), cy=int(cy), w=int(ww), h=int(hh), theta=float(theta), score=score))
            best_conf = max(best_conf, score)
        return obbs, best_conf


class YoloV11Detector(BaseDetector):
    """
    如需接入真实 YOLO（例如 ultralytics 的 yolo11-obb），在此类中完成:
      1) 模型加载（__init__）
      2) 在 ROI 上截取后推理，解析为 OBB 列表
    """

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        self.model = None
        self.ready = False
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path or "yolo11n-obb.pt")
            self.ready = True
            pass
        except Exception as e:
            print("YOLO Loaded failed, back to default detector:", e)

    def infer_obb(self, crop: np.ndarray, target: int = -1) -> \
            Tuple[List[Dict[str, Any]], float]:
        if not self.ready:
            return BaseDetector.infer_obb(self, crop)
        # TODO: 截取 ROI，送入 YOLO OBB 推理，组装 obbs
        # 伪代码:
        results = self.model(crop, conf=0.25, iou=0.5)
        obbs = parse_yolo_obb(results, target)  # 转为 [{'cx','cy','w','h','theta','score'}, ...]
        best_conf = max([o['score'] for o in obbs], default=0.0)
        return obbs, best_conf
        # return BaseDetector.infer_obb(self, image_bgr, roi_xyxy)


def parse_yolo_obb(results, target: int = -1):
    """
    Convert YOLO OBB results to list of dicts:
    [{'cx','cy','w','h','theta','score'}, ...]
    """
    obbs = []
    for r in results:  # each image
        if not hasattr(r, "obb"):  # safety
            continue
        # r.obb is an ultralytics.OBB object with xywhr, conf, cls
        xywhr = r.obb.xywhr.cpu().numpy()  # (N,5) [x,y,w,h,theta(rad)]
        confs = r.obb.conf.cpu().numpy()  # (N,)
        cls = r.obb.cls.cpu().numpy().astype(int)  # (N,)
        for (cx, cy, w, h, theta), score, cl in zip(xywhr, confs, cls):
            if target < 0 or target == cl:
                obbs.append({
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "theta": float(theta * 180 / math.pi),  # convert rad→deg
                    "score": float(score),
                    "class": cl
                })
    return obbs
