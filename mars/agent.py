import json
import os.path
import pickle
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple

# ----------------------------
# 强化学习控制器（简化 Q-Learning）
# ----------------------------
ACTIONS = ["up", "down", "left", "right", "zoom_in", "zoom_out"]


class Agent(ABC):

    @abstractmethod
    def select_action(self, i, j, scale_bin, actions) -> str:
        pass

    @abstractmethod
    def update(self, s, a, r, s2):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def clear(self):
        pass


class RLQtableAgent(Agent):

    def __init__(self, grid_shape=(10, 10), eps=0.15, alpha=0.5, gamma=0.9, training=False, load: bool = False,
                 model: str = 'qtable.pkl'):
        self.training = training
        self.H, self.W = grid_shape
        self.model: Dict[Tuple[int, int, int, str], float] = {}  # (i,j,scale_bin,action) -> Q
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        if load and os.path.exists(model):
            self.load(model)

    def _key(self, i, j, scale_bin, a):
        return (i, j, scale_bin, a)

    def select_action(self, i, j, scale_bin, actions) -> str:
        # ε-greedy
        if random.random() < self.eps:
            return random.choice(actions)
        # 选 Q 值最大的动作；若都未见过，则启发式优先未访问区域
        best_a, best_q = None, -1e9
        for a in actions:
            q = self.model.get(self._key(i, j, scale_bin, a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a or random.choice(actions)

    def update(self, s, a, r, s2):
        if not self.training:
            return
        (i, j, scale_bin, actions) = s
        (i2, j2, scale_bin2, actions2) = s2
        key = self._key(i, j, scale_bin, a)
        q = self.model.get(key, 0.0)
        # 目标 = r + gamma * max_a' Q(s',a')
        max_next = max([self.model.get(self._key(i2, j2, scale_bin2, ap), 0.0) for ap in actions2])
        new_q = q + self.alpha * (r + self.gamma * max_next - q)
        self.model[key] = new_q

    def save(self, filename: str = 'qtable.pkl'):
        """
        保存Q表到文件
        参数:
            filename: 文件名
            format: 保存格式，可选 'pickle' 或 'json'
        """
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'eps': self.eps,
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'grid_shape': (self.H, self.W)
                }, f)
        elif filename.endswith('.json'):
            # 将Q表转换为可JSON序列化的格式
            qtable_serializable = {
                f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in self.model.items()
            }
            data = {
                'Q': qtable_serializable,
                'eps': self.eps,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'grid_shape': [self.H, self.W]
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError("format must be 'pickle' or 'json'")
        print(f"Q表已保存到 {filename}.")

    def load(self, filename: str = 'qtable.pkl'):
        """
        从文件加载Q表

        参数:
            filename: 文件名
            format: 文件格式，可选 'auto', 'pickle', 或 'json'
        """
        try:
            if filename.endswith('.pkl'):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.eps = data.get('eps', self.eps)
                    self.alpha = data.get('alpha', self.alpha)
                    self.gamma = data.get('gamma', self.gamma)
                    if 'grid_shape' in data:
                        self.H, self.W = data['grid_shape']
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    # 将字符串键转换回元组键
                    self.model = {}
                    for key_str, value in data['model'].items():
                        parts = key_str.split(',')
                        key_tuple = (int(parts[0]), int(parts[1]), int(parts[2]), parts[3])
                        self.model[key_tuple] = value
                    self.eps = data.get('eps', self.eps)
                    self.alpha = data.get('alpha', self.alpha)
                    self.gamma = data.get('gamma', self.gamma)
                    if 'grid_shape' in data:
                        self.H, self.W = data['grid_shape']
            print(f"Q表已从 {filename} 加载，共 {len(self.model)} 个状态-动作对")

        except FileNotFoundError:
            print(f"文件 {filename} 不存在，使用空的Q表")
        except Exception as e:
            print(f"加载Q表时出错: {e}")
            # 保持当前Q表不变

    def clear(self):
        """清空Q表"""
        self.model.clear()
        print("Q表已清空")
