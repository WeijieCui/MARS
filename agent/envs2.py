import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class Env(gym.Env):
    def __init__(self, base_image, vm, target_classes, ground_truth, history_length=5, max_steps=20):
        """
        Custom environment for active object detection

        Args:
            base_image (np.ndarray): Full-resolution image (H, W, 3)
            vm: Vision model with .predict() method
            target_classes (list): Target class IDs
            ground_truth (list): Ground truth objects [class_id, x, y, w, h]
            history_length (int): Number of steps to keep in history
            max_steps (int): Maximum steps per episode
        """
        super().__init__()
        self.base_image = base_image
        self.height, self.width = base_image.shape[:2]
        self.vm = vm
        self.target_classes = target_classes
        self.ground_truth = ground_truth
        self.H = history_length
        self.max_steps = max_steps

        # Convert ground truth to more usable format
        self.gt_objects = []
        for idx, obj in enumerate(ground_truth):
            class_id, x, y, w, h = obj
            self.gt_objects.append({
                'id': idx,
                'class_id': class_id,
                'bbox': [x, y, w, h]
            })

        # Action space: (x, y, scale) all in [0, 1]
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        # Observation space: History vector + current patch
        n_classes = len(target_classes)
        history_size = self.H * (3 + 2 * n_classes)  # (action + class summary)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(history_size,), dtype=np.float32
            ),
            'current_patch': gym.spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3), dtype=np.uint8
            )
        })

        # Initialize in reset()
        self.step_count = None
        self.detected_objects = None
        self.history = None

    def reset(self):
        """Reset environment to initial state"""
        self.step_count = 0
        self.detected_objects = set()  # Track detected ground truth objects
        self.history = deque(maxlen=self.H)  # Store (action, class_summary)

        # Create initial state
        history_vector = self._get_history_vector()
        current_patch = np.zeros((64, 64, 3), dtype=np.uint8)  # Blank patch

        return {'history': history_vector, 'current_patch': current_patch}

    def step(self, action):
        """Execute one action step in the environment"""
        # Unpack and denormalize action
        x, y, scale = action
        center_x = int(x * self.width)
        center_y = int(y * self.height)

        # Calculate patch size (scale: 0.1-1.0 of min dimension)
        min_dim = min(self.width, self.height)
        patch_size = int(scale * min_dim * 0.9 + 0.1 * min_dim)

        # Extract patch
        patch = self._extract_patch(center_x, center_y, patch_size)

        # Get detections from vision model
        detections = self.vm.predict(patch)

        # Calculate reward
        new_detections = self._match_detections(detections)
        recall_gain = len(new_detections) / max(1, len(self.gt_objects))
        step_penalty = 0.01
        reward = recall_gain - step_penalty
        self.detected_objects.update(new_detections)

        # Update history
        class_summary = self._compute_class_summary(detections)
        self.history.append((action, class_summary))

        # Prepare next state
        resized_patch = cv2.resize(patch, (64, 64))
        history_vector = self._get_history_vector()
        next_state = {
            'history': history_vector,
            'current_patch': resized_patch
        }

        # Update step count and check termination
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return next_state, reward, done, False, {}

    def _extract_patch(self, center_x, center_y, patch_size):
        """Extract image patch with boundary checks"""
        half_size = patch_size // 2
        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(self.width, center_x + half_size)
        y1 = min(self.height, center_y + half_size)

        # Handle edge cases
        if x1 - x0 < 1 or y1 - y0 < 1:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return self.base_image[y0:y1, x0:x1]

    def _compute_class_summary(self, detections):
        """Create class summary vector from detections"""
        summary = np.zeros(2 * len(self.target_classes), dtype=np.float32)

        # Group detections by class
        class_counts = {cls: [] for cls in self.target_classes}

        for det in detections:
            if len(det) < 6:  # Ensure valid detection
                continue
            *_, conf, cls_id = det[:6]
            if cls_id in class_counts:
                class_counts[cls_id].append(conf)

        # Fill summary vector: [count, avg_conf] for each class
        for i, cls_id in enumerate(self.target_classes):
            confs = class_counts[cls_id]
            count = len(confs)
            avg_conf = np.mean(confs) if count > 0 else 0.0
            summary[2 * i] = count
            summary[2 * i + 1] = avg_conf

        return summary

    def _match_detections(self, detections):
        """Match detections to ground truth objects using IoU"""
        new_detections = set()
        matched_gt = set()

        for det in detections:
            if len(det) < 6:
                continue
            x, y, w, h, conf, cls_id = det[:6]
            det_bbox = [x, y, w, h]

            for gt in self.gt_objects:
                if gt['id'] in matched_gt or gt['class_id'] != cls_id:
                    continue

                iou = self._calculate_iou(det_bbox, gt['bbox'])
                if iou > 0.5:  # IoU threshold
                    new_detections.add(gt['id'])
                    matched_gt.add(gt['id'])
                    break

        return new_detections - self.detected_objects

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate coordinates
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        # Calculate area of intersection
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _get_history_vector(self):
        """Convert history deque to fixed-size vector"""
        vec_size = 3 + 2 * len(self.target_classes)
        history_vec = np.zeros(self.H * vec_size, dtype=np.float32)

        # Fill from oldest to newest
        for i, (action, summary) in enumerate(self.history):
            start_idx = i * vec_size
            history_vec[start_idx:start_idx + 3] = action
            history_vec[start_idx + 3:start_idx + vec_size] = summary

        return history_vec


# if __name__ == '__main__':
#     # Example initialization
#     env = Env(
#         base_image=your_image_array,
#         vm=your_vision_model,
#         target_classes=[0, 1, 2],  # Example class IDs
#         ground_truth=[
#             [0, 100, 150, 50, 50],  # [class, x, y, w, h]
#             [1, 300, 200, 60, 60]
#         ],
#         history_length=5,
#         max_steps=20
#     )
#
#     # Standard gym interface
#     obs = env.reset()
#     action = env.action_space.sample()
#     next_obs, reward, done, _, _ = env.step(action)
