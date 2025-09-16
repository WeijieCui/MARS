import logging
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from rl_model import RLQtableModel
from torch.utils.data import Dataset

from detector import YoloV11Detector
from env import SearchEnv

os.environ['GLOG_minloglevel'] = '3'
logging.disable(logging.CRITICAL)


class DotaRawDataset(Dataset):
    """
    Loading DOTA PyTorch Dataset
    """

    def __init__(
            self,
            image_dir,
            label_dir,
            transform=None,
            joint_transform=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _parse_label_file(self, label_path):
        bboxes = []
        classes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()[2:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                points = list(map(float, parts[:8]))
                cls = parts[8]
                bbox = [(points[i], points[i + 1]) for i in range(0, 8, 2)]
                bboxes.append(bbox)
                classes.append(cls)
        return bboxes, classes

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.txt')
        img_bgr = cv2.imread(image_path)
        if os.path.exists(label_path):
            bboxes, classes = self._parse_label_file(label_path)
        else:
            bboxes, classes = [], []
        return img_bgr, {'boxes': bboxes, 'labels': classes}


def train():
    episodes = 100
    log_interval = 5
    MAX_NUM_EXPLORING_STEP = 20
    rl_model = RLQtableModel(save=True, load=True)
    base_data_dir = '..\\..\\data\\train' if os.path.exists('..\\..\\data\\train') else '..\\data\\train'
    train_dataset = DotaRawDataset(
        image_dir=os.path.join(base_data_dir, 'images'),
        label_dir=os.path.join(base_data_dir, 'labelTxt')
    )

    env = SearchEnv()
    env.set_detector(YoloV11Detector(device='CPU'))

    # data for performance figures
    episode_rewards = []
    episode_vehicles_found = []
    episode_steps = []
    moving_avg_rewards = []
    moving_avg_found = []

    for episode in range(episodes):
        vehicles_found_this_episode = 0
        total_reward = 0
        image_steps = []
        count = 0
        start = time.time()
        for image, target in train_dataset:
            count += 1
            env.set_image(image)
            env.set_references(target)
            status, reward, obbs, new_obbs, window = env.reset()
            image_reward = 0
            done = False
            step = 0
            steps_count = 0
            obbs = []
            for step in range(MAX_NUM_EXPLORING_STEP):
                # select action
                action = rl_model.select_action(*status)
                if not action:
                    break
                # act
                next_status, reward, obbs, new_obbs, window = env.step(action)
                # learn
                rl_model.update(status, action, reward, next_status)
                status = next_status
                image_reward += reward
            # record this round
            steps_count += step
            total_reward += image_reward
            episode_steps.append(steps_count)
            vehicles_found_this_episode += len(obbs)
            if count % 10 == 0:
                print("processed {} images, time: {:.2f} seconds.".format(count, time.time() - start))
        episode_rewards.append(total_reward)
        episode_vehicles_found.append(vehicles_found_this_episode)

        # Calculate a sliding average (window size 100) to make it easier to see trends
        moving_avg_r = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        moving_avg_rewards.append(moving_avg_r)
        moving_avg_f = np.mean(episode_vehicles_found[-100:]) if len(episode_vehicles_found) >= 100 else np.mean(
            episode_vehicles_found)
        moving_avg_found.append(moving_avg_f)

        # Print logs regularly
        print(f"Episode {episode:4d}/{episodes} | "
              f"Reward: {total_reward:6.1f} | "
              f"Vehicles Found: {vehicles_found_this_episode:2d} | "
              f"Steps: {episode_steps} | "
              f"Avg Reward (MA100): {moving_avg_r:6.1f} | "
              f"Avg Found (MA100): {moving_avg_f:4.1f}")
        rl_model.save('qtable-{}.pkl'.format(episode))
    # Draw a chart after training
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episode_vehicles_found, label='Per Episode', alpha=0.3)
    plt.plot(moving_avg_found, label='Moving Avg (100)', linewidth=2)
    plt.axhline(y=100, color='r', linestyle='--', label='True Count')
    plt.xlabel('Episode')
    plt.ylabel('Vehicles Found')
    plt.title('Performance: Vehicles Found per Episode')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episode_rewards, label='Per Episode', alpha=0.3)
    plt.plot(moving_avg_rewards, label='Moving Avg (100)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Performance: Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.title('Efficiency: Steps per Episode')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


if __name__ == '__main__':
    train()
