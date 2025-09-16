from utils import merge_bounding_box


def test_merge_bounding_box():
    centers = [(30, 30), (50, 30),
               (50, 50), (70, 50),
               (100, 100)]
    all_boxes = [{
        'cx': x,
        'cy': y,
        'w': 10,
        'h': 10,
        'theta': 0.3
    } for x, y in centers]
    boxes = [{
        'cx': x,
        'cy': y,
        'w': 8,
        'h': 10,
        'theta': 0.2
    } for x, y in centers[:3]]
    merged_boxes = merge_bounding_box(all_boxes, boxes)
    assert max(len(all_boxes), len(boxes)) <= len(merged_boxes) <= len(all_boxes) + len(boxes)
    assert len(all_boxes) == len(merged_boxes)
