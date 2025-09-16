import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


# 假设你已经有了YOLO模型和QTable类
# from yolov11 import YOLOv11Detector
# from qlearning_agent import QLearningAgent

class VehicleCountingEnv:
    def __init__(self, large_image, true_vehicle_count):
        self.large_image = large_image
        self.true_vehicle_count = true_vehicle_count
        self.current_position = (0, 0)  # 智能体当前视角的左上角坐标
        self.view_size = (224, 224)  # 智能体每次看到的局部图像大小
        self.visited_cells = set()  # 记录访问过的网格位置（离散化后），用于计算探索覆盖率
        self.found_vehicles = set()  # 使用set存储唯一车辆标识（例如用bbox的中心点坐标哈希），防止重复计数
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        # 重置环境状态
        self.current_position = (0, 0)
        self.visited_cells.clear()
        self.found_vehicles.clear()
        self.step_count = 0
        cell_x = self.current_position[0] // self.view_size[0]
        cell_y = self.current_position[1] // self.view_size[1]
        self.visited_cells.add((cell_x, cell_y))
        # 获取初始观察
        observation = self._get_observation()
        return self._get_state_representation(observation)

    def step(self, action):
        """
        执行动作 (0:上, 1:右, 2:下, 3:左)
        """
        self.step_count += 1
        reward = -0.1  # 每一步的小惩罚，鼓励高效
        done = False

        # 1. 移动智能体
        x, y = self.current_position
        step_size = 50  # 移动步长
        if action == 0:
            y = max(0, y - step_size)
        elif action == 1:
            x = min(self.large_image.width - self.view_size[0], x + step_size)
        elif action == 2:
            y = min(self.large_image.height - self.view_size[1], y + step_size)
        elif action == 3:
            x = max(0, x - step_size)
        self.current_position = (x, y)

        # 2. 记录访问过的网格（离散化）
        cell_x = x // self.view_size[0]
        cell_y = y // self.view_size[1]
        cell_key = (cell_x, cell_y)
        if cell_key in self.visited_cells:
            reward -= 0.5  # 重复访问区域的惩罚
        else:
            self.visited_cells.add(cell_key)

        # 3. 获取新观察，运行YOLO
        observation = self._get_observation()
        detections = yolo_detector.detect(observation)
        current_frame_vehicles = set()

        # 4. 处理检测结果，计算奖励
        for det in detections:
            if det['class'] == 'car':  # 根据你的YOLO类别调整
                # 创建一个唯一标识符，例如 bounding box 的中心点
                cx = (det['bbox'][0] + det['bbox'][2]) / 2 + x  # 转换为全局坐标
                cy = (det['bbox'][1] + det['bbox'][3]) / 2 + y
                vehicle_id = f"{int(cx)}_{int(cy)}"
                current_frame_vehicles.add(vehicle_id)

                if vehicle_id not in self.found_vehicles:
                    # 发现新车，大奖励！
                    self.found_vehicles.add(vehicle_id)
                    reward += 20

        # 5. 检查终止条件
        if self.step_count >= self.max_steps:
            done = True
        # 可选：如果找到了所有车，也可以提前结束
        # if len(self.found_vehicles) >= self.true_vehicle_count:
        #    reward += 100
        #    done = True

        next_state = self._get_state_representation(observation)
        info = {
            'detections': detections,
            'found_vehicles': len(self.found_vehicles),
            'steps': self.step_count
        }
        return next_state, reward, done, info

    def _get_observation(self):
        # 从大图上裁剪出当前视角的图像
        x, y = self.current_position
        w, h = self.view_size
        observation = self.large_image[y:y + h, x:x + w]
        return observation

    def _get_state_representation(self, observation):
        # 将YOLO的检测结果（如车辆位置、数量）和智能体自身位置编码成Q表的状态
        # 这是一个关键且需要你精心设计的部分
        # 例如：可以将图像划分为4x4网格，统计每个网格内的车辆数，得到一个16维的状态向量
        # 或者使用更简单的：当前视野内车辆数量 + 当前智能体的归一化坐标
        state_rep = ...  # 你的状态设计逻辑
        return state_rep


# 初始化环境和智能体
env = VehicleCountingEnv(large_image, true_vehicle_count=50)
agent = QLearningAgent(state_size=..., action_size=4)

# 训练循环 + 跟踪指标
episodes = 1000
log_interval = 20

# 用于绘图的列表
episode_rewards = []
episode_vehicles_found = []
episode_steps = []
moving_avg_rewards = []
moving_avg_found = []

for episode in range(episodes):
    status, reward, obbs, new_obbs, window = env.reset()
    total_reward = 0
    done = False
    info = {}

    while not done:
        # 选择动作
        action = agent.act(state)
        # 执行动作
        next_state, reward, done, info = env.step(action)
        # 学习
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    # 记录本轮指标
    episode_rewards.append(total_reward)
    vehicles_found_this_episode = info['found_vehicles']
    episode_vehicles_found.append(vehicles_found_this_episode)
    episode_steps.append(info['steps'])

    # 计算滑动平均（窗口大小为100）以便更容易看到趋势
    moving_avg_r = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    moving_avg_rewards.append(moving_avg_r)
    moving_avg_f = np.mean(episode_vehicles_found[-100:]) if len(episode_vehicles_found) >= 100 else np.mean(
        episode_vehicles_found)
    moving_avg_found.append(moving_avg_f)

    # 定期打印日志
    if episode % log_interval == 0:
        print(f"Episode {episode:4d}/{episodes} | "
              f"Reward: {total_reward:6.1f} | "
              f"Vehicles Found: {vehicles_found_this_episode:2d} | "
              f"Steps: {info['steps']:3d} | "
              f"Avg Reward (MA100): {moving_avg_r:6.1f} | "
              f"Avg Found (MA100): {moving_avg_f:4.1f}")

# 训练结束后绘制图表
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(episode_vehicles_found, label='Per Episode', alpha=0.3)
plt.plot(moving_avg_found, label='Moving Avg (100)', linewidth=2)
plt.axhline(y=env.true_vehicle_count, color='r', linestyle='--', label='True Count')
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

# ... 你还可以添加Loss等图 ...

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()
