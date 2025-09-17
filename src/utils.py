from typing import List, Dict, Any

from shapely.geometry import Polygon
import math


def merge_bounding_box(all_boxes: List[Dict[str, Any]], boxes: List[Dict[str, Any]]):
    new_joins = []
    for nb in boxes:
        merged = False
        box1 = (nb['cx'], nb['cy'], nb['w'], nb['h'], nb['theta'] * 180)
        for ob in all_boxes:
            if ob['class'] == nb['class']:
                box2 = (ob['cx'], ob['cy'], ob['w'], ob['h'], ob['theta'] * 180)
                iou = obb_iou(box1, box2)
                if iou > 0.15:
                    merged = True
                    break
        if not merged:
            new_joins.append(nb)
    return new_joins


def obb_to_polygon(cx, cy, w, h, angle_deg):
    """
    Convert bounding box (cx, cy, w, h, angle) to shapely Polygon
    cx, cy: center
    w, h: width, height
    angle_deg: Rotation angle (degrees, counterclockwise)
    """
    return Polygon(obb_to_vertices(cx, cy, w, h, angle_deg))

def obb_to_vertices(cx, cy, w, h, angle_deg):
    """
    Convert bounding box (cx, cy, w, h, angle) to shapely Polygon
    cx, cy: center
    w, h: width, height
    angle_deg: Rotation angle (degrees, counterclockwise)
    """
    angle = math.radians(angle_deg)
    dx = w / 2
    dy = h / 2

    # Four vertices when not rotated (relative to the center)
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy)
    ]

    # Rotation + Movement
    rotated = []
    for x, y in corners:
        xr = int(x * math.cos(angle) - y * math.sin(angle))
        yr = int(x * math.sin(angle) + y * math.cos(angle))
        rotated.append((cx + xr, cy + yr))
    return rotated

def obb_iou(box1, box2):
    """Calculate the IoU of two boxes"""
    poly1 = obb_to_polygon(*box1)
    poly2 = obb_to_polygon(*box2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def weighted_fusion(boxes, scores):
    """
    Confidence-weighted merging of multiple OBBs
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
