import numpy as np
from enum import Enum

from agent.env import Env


# 定义动作枚举
class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# 自定义环境类
class GridWorldEnv(Env):
    """
    自定义网格世界环境
    4x4网格，智能体从(0,0)出发，目标到达(3,3)
    """

    def __init__(self, size=4):
        self.size = size  # 网格大小
        self.start_pos = (0, 0)  # 起始位置
        self.goal_pos = (size - 1, size - 1)  # 目标位置
        self.agent_pos = None  # 智能体当前位置

        # 定义动作空间 (4个方向)
        self.action_space = list(Action)

        # 定义观察空间 (位置坐标)
        self.observation_space = (size, size)

        # 渲染相关
        self.grid = None

    def reset(self):
        """重置环境，返回初始状态"""
        self.agent_pos = self.start_pos
        self._update_grid()
        return self.agent_pos

    def step(self, action):
        """执行一个动作，返回(next_state, reward, done, info)"""
        x, y = self.agent_pos

        # 根据动作移动智能体
        if action == Action.UP:
            x = max(0, x - 1)
        elif action == Action.RIGHT:
            y = min(self.size - 1, y + 1)
        elif action == Action.DOWN:
            x = min(self.size - 1, x + 1)
        elif action == Action.LEFT:
            y = max(0, y - 1)

        self.agent_pos = (x, y)
        self._update_grid()

        # 检查是否到达目标
        done = (self.agent_pos == self.goal_pos)
        reward = 10 if done else -0.1  # 到达目标获得正奖励，否则小惩罚

        return self.agent_pos, reward, done, {}

    def _update_grid(self):
        """更新网格状态用于渲染"""
        self.grid = np.zeros((self.size, self.size), dtype=str)
        self.grid[:, :] = '.'  # 空位置

        # 设置智能体和目标位置
        self.grid[self.agent_pos] = 'A'  # 智能体
        self.grid[self.goal_pos] = 'G'  # 目标

    def render(self):
        """渲染当前环境状态"""
        if self.grid is None:
            self._update_grid()

        print(f"当前状态 (智能体位置: {self.agent_pos})")
        for row in self.grid:
            print(' '.join(row))
        print()

    def close(self):
        """关闭环境，清理资源"""
        self.grid = None
        print("环境已关闭")


# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = GridWorldEnv(size=4)

    # 重置环境
    state = env.reset()
    print("初始状态:", state)
    env.render()

    # 执行一些动作
    actions = [Action.RIGHT, Action.RIGHT, Action.DOWN, Action.DOWN, Action.RIGHT, Action.DOWN]

    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)

        print(f"步骤 {i + 1}:")
        print(f"动作: {action.name}")
        print(f"奖励: {reward}, 完成: {done}")
        env.render()

        if done:
            print("成功到达目标！")
            break

    # 关闭环境
    env.close()
