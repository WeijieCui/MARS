import os

import cv2

from detector import YoloV11Detector
from dota_dataset import DOTA_LABELS, IMG_SUFFIX
from env import SearchEnv
from rl_model import RLQtableModel
from utils import obb_to_vertices


class MARSAgent:
    def __init__(self, rl_model, device: str = 'cpu'):
        self.search_env = SearchEnv()
        self.search_env.set_detector(YoloV11Detector(device=device))
        self.rl_model = rl_model

    def visual_search(self, image, target='', max_step: int = 10, learn: bool = False):
        self.search_env.set_image(image)
        self.search_env.set_references(target)
        status, reward, obbs, new_obbs, window = self.search_env.reset()
        image_reward = 0
        step = 0
        obbs = []
        for step in range(max_step):
            # select action
            action = self.rl_model.select_action(*status)
            if not action:
                break
            # act
            next_status, reward, obbs, new_obbs, window = self.search_env.step(action)
            # learn
            if learn:
                self.rl_model.update(status, action, reward, next_status)
            status = next_status
            image_reward += reward
        return step, image_reward, obbs

    def batch_visual_search(self, img_dir, result_dir, max_step: int = 20):
        results = {label: [] for label in range(15)}
        label_dict = {label: name for name, label in DOTA_LABELS.items()}
        for count, img_name in enumerate([f for f in os.listdir(img_dir) if f.endswith(IMG_SUFFIX)]):
            step, image_reward, obbs = self.visual_search(
                image=cv2.imread(os.path.join(img_dir, img_name)),
                max_step=max_step,
            )
            for obb in obbs:
                vertices = obb_to_vertices(obb['cx'], obb['cy'], obb['w'], obb['h'], obb['theta'] * 180)
                vertice_expended = []
                for x, y in vertices:
                    vertice_expended.append(int(x))
                    vertice_expended.append(int(y))
                results[obb['class']].append(
                    [img_name.split('.')[0], obb['score'], *vertice_expended])
            print(f'\rprocessed {count + 1} images.', end='', flush=True)
        # write prediction to file: imgname score x1 y1 x2 y2 x3 y3 x4 y4 plane.txt, storage-tank.txt
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            for label, targets in results.items():
                with open(os.path.join(result_dir, label_dict[label] + '.txt'), 'w') as file:
                    for target in targets:
                        file.write(' '.join(str(i) for i in target) + '\n')
        return results
