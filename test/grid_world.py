
import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.grid_size = 4
        self.action_space = spaces.Discrete(4)  # 0:上, 1:右, 2:下, 3:左
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)

        # 定义目标位置和障碍物
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (2, 2)]
        self.start = (0, 0)
        self.agent_pos = None

        # 渲染设置
        self.render_mode = render_mode

        # 随机数生成器
        self.np_random = None

    def reset(self, seed=None, options=None):
        # 初始化随机数生成器
        super().reset(seed=seed)
        self.np_random, seed = seeding.np_random(seed)

        # 重置代理位置
        self.agent_pos = np.array(self.start)

        if self.render_mode == 'human':
            self._render_frame()

        # 返回元组 (observation, info)
        return self.agent_pos.copy(), {}

    def step(self, action):
        # 定义动作对应的移动
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        move = moves[action]

        # 计算新位置
        new_pos = self.agent_pos + move

        # 边界检查
        if (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size):
            # 障碍物检查
            if tuple(new_pos) not in self.obstacles:
                self.agent_pos = new_pos

        # 检查是否到达目标
        terminated = np.array_equal(self.agent_pos, self.goal)
        truncated = False  # 我们的环境没有时间限制
        reward = 10 if terminated else -0.1  # 到达目标奖励，否则小惩罚

        if self.render_mode == 'human':
            self._render_frame()

        # 返回新API格式: (obs, reward, terminated, truncated, info)
        return self.agent_pos.copy(), reward, terminated, truncated, {}

    def render(self):
        """实现渲染方法"""
        if self.render_mode is None:
            # 环境没有设置渲染模式
            return None

        if self.render_mode == 'human':
            # 在控制台打印网格
            self._render_frame()
            return None
        elif self.render_mode == 'ansi':
            # 返回ANSI字符串表示
            return self._render_ansi()
        else:
            raise NotImplementedError(f"渲染模式 '{self.render_mode}' 未实现")

    def _render_frame(self):
        """渲染到控制台（人类可读）"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 标记目标
        grid[self.goal[0]][self.goal[1]] = 'G'

        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        # 打印网格
        grid_str = "\n".join([" ".join(row) for row in grid])
        print(grid_str)
        print()

    def _render_ansi(self):
        """返回环境的ANSI字符串表示"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 标记目标
        grid[self.goal[0]][self.goal[1]] = 'G'

        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        # 创建网格字符串
        grid_str = "\n".join([" ".join(row) for row in grid])
        return grid_str

    def close(self):
        """清理资源"""
        pass