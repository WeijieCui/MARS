# evaluation/metrics.py
def evaluate_mars(test_set):
    results = []
    for img, question, gt in test_set:
        answer, detections = agent.execute_query(img, question)
        # Grounding accuracy
        iou = calculate_polygon_iou(detections, gt['polygons'])
        # VQA accuracy
        vqa_acc = exact_match(answer, gt['answer'])
        results.append((iou, vqa_acc))

    mean_iou = np.mean([r[0] for r in results])
    mean_acc = np.mean([r[1] for r in results])
    return mean_iou, mean_acc