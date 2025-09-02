import random
from typing import Dict, Tuple

# ----------------------------
# 强化学习控制器（简化 Q-Learning）
# ----------------------------
ACTIONS = ["up", "down", "left", "right", "zoom_in", "zoom_out"]


class RLAgent:
    def __init__(self, grid_shape=(10, 10), eps=0.15, alpha=0.5, gamma=0.9, training=False):
        self.H, self.W = grid_shape
        self.Q: Dict[Tuple[int, int, int, str], float] = {}  # (i,j,scale_bin,action) -> Q
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.training = training

    def _key(self, i, j, scale_bin, a):
        return (i, j, scale_bin, a)

    def select_action(self, i, j, scale_bin, actions) -> str:
        # ε-greedy
        if random.random() < self.eps:
            return random.choice(actions)
        # 选 Q 值最大的动作；若都未见过，则启发式优先未访问区域
        best_a, best_q = None, -1e9
        for a in actions:
            q = self.Q.get(self._key(i, j, scale_bin, a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a or random.choice(actions)

    def update(self, s, a, r, s2):
        if not self.training:
            return
        (i, j, scale_bin, actions) = s
        (i2, j2, scale_bin2, actions2) = s2
        key = self._key(i, j, scale_bin, a)
        q = self.Q.get(key, 0.0)
        # 目标 = r + gamma * max_a' Q(s',a')
        max_next = max([self.Q.get(self._key(i2, j2, scale_bin2, ap), 0.0) for ap in ACTIONS])
        new_q = q + self.alpha * (r + self.gamma * max_next - q)
        self.Q[key] = new_q
