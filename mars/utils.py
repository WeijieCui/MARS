from typing import List, Dict, Any

from shapely.geometry import Polygon
import math


def merge_bounding_box(all_boxes: List[Dict[str, Any]], boxes: List[Dict[str, Any]]):
    new_joins = []
    merged_boxes = []
    for nb in boxes:
        merged = False
        box1 = (nb['cx'], nb['cy'], nb['w'], nb['h'], nb['theta'] * 180)
        for ob in all_boxes:
            box2 = (ob['cx'], ob['cy'], ob['w'], ob['h'], ob['theta'] * 180)
            iou = obb_iou(box1, box2)
            if iou > 0.2:
                merged_boxes.append((ob, nb))
                merged = True
                break
        if not merged:
            new_joins.append(nb)
    all_boxes.extend(new_joins)
    for ob, nb in merged_boxes:
        box1 = (nb['cx'], nb['cy'], nb['w'], nb['h'], nb['theta'] * 180)
        box2 = (ob['cx'], ob['cy'], ob['w'], ob['h'], ob['theta'] * 180)
        merged_box = weighted_fusion([box1, box2], [0.9, 0.8])
        ob['cx'] = merged_box[0]
        ob['cy'] = merged_box[1]
        ob['w'] = merged_box[2]
        ob['h'] = merged_box[3]
        ob['theta'] = merged_box[4] / 180


def obb_to_polygon(cx, cy, w, h, angle_deg):
    """
    将旋转框 (cx, cy, w, h, angle) 转换为 shapely Polygon
    cx, cy: 中心点
    w, h: 宽高
    angle_deg: 旋转角度 (度数制，逆时针)
    """
    angle = math.radians(angle_deg)
    dx = w / 2
    dy = h / 2

    # 未旋转时的四个顶点 (相对中心)
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy)
    ]

    # 旋转 + 平移
    rotated = []
    for x, y in corners:
        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)
        rotated.append((cx + xr, cy + yr))

    return Polygon(rotated)


def obb_iou(box1, box2):
    """计算两个 OBB 的 IoU"""
    poly1 = obb_to_polygon(*box1)
    poly2 = obb_to_polygon(*box2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def weighted_fusion(boxes, scores):
    """
    对多个 OBB 进行置信度加权合并
    boxes: [(cx, cy, w, h, angle), ...]
    scores: [score1, score2, ...]
    """
    total_score = sum(scores)
    cx = sum(b[0] * s for b, s in zip(boxes, scores)) / total_score
    cy = sum(b[1] * s for b, s in zip(boxes, scores)) / total_score
    w = sum(b[2] * s for b, s in zip(boxes, scores)) / total_score
    h = sum(b[3] * s for b, s in zip(boxes, scores)) / total_score
    angle = sum(b[4] * s for b, s in zip(boxes, scores)) / total_score
    return (cx, cy, w, h, angle)
