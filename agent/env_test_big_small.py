import gym
from gym import spaces
import warnings
import numpy as np

# 忽略特定警告
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        message='`np.bool8` is a deprecated alias')
grid_size = 9


class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(6)  # 0:上, 1:右, 2:下, 3:左, 4: 放大, 5: 缩小
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)

        # 定义目标位置
        self.goal = (5, 5)
        self.goal_window = 3
        self.start = (3, 3)
        self.agent_pos = None
        self.window = 3

    def reset(self):
        self.agent_pos = np.array(self.start)
        return self.agent_pos.copy()

    def update_position(self, action):
        # 定义动作对应的移动
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        # 计算新位置
        if action < 4:
            move = np.dot(moves[action], self.window)
            new_pos = self.agent_pos + move
            # 边界检查
            if (0 <= new_pos[0] < self.grid_size - self.window and
                    0 <= new_pos[1] < self.grid_size - self.window):
                self.agent_pos = new_pos
        elif action == 4:
            if self.window < self.grid_size:
                window = self.window + 2
                new_pos = self.agent_pos + (-1, -1)
                if (0 <= new_pos[0] < self.grid_size - window and
                        0 <= new_pos[0] < self.grid_size - window):
                    self.window = window
                    self.agent_pos = new_pos
        else:
            if self.window > 1:
                self.window -= 2
                self.agent_pos += (1, 1)

    def step(self, action):
        self.update_position(action)
        # 检查是否到达目标
        done = np.array_equal(self.agent_pos, self.goal) and self.goal_window == self.window
        reward = 10 if done else -0.1  # 到达目标奖励，否则小惩罚
        return self.agent_pos.copy(), reward, done, {}

    def render(self, mode='human'):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # 标记目标
        grid[self.goal[0]][self.goal[1]] = 'G' + str(self.goal_window)
        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A' + str(self.window)
        # 打印网格
        print("\n".join([" ".join(row) for row in grid]))
        print("\n")


# 注册自定义环境
gym.register(id='GridWorld-v0', entry_point=GridWorldEnv)

# 创建环境
env = gym.make('GridWorld-v0')

# Q-learning 参数
q_table = np.zeros((grid_size, grid_size, 6))  # [x, y, action]
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
episodes = 1000

# 训练循环
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # ε-贪婪策略
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x, y = state
            action = np.argmax(q_table[x, y])
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        # 更新Q值
        x, y = state
        next_x, next_y = next_state
        # Q(s,a) = Q(s,a) + α[r + γmaxQ(s',a') - Q(s,a)]
        q_table[x, y, action] += learning_rate * (
                reward +
                discount_factor * np.max(q_table[next_x, next_y]) -
                q_table[x, y, action]
        )
        state = next_state

print("训练完成！")
# 训练循环后添加保存代码
np.save("q_table.npy", q_table)
print("Q表已保存到 q_table.npy")
# 加载环境
# gym.register(id='GridWorld-v0', entry_point=GridWorldEnv)
# env = gym.make('GridWorld-v0')
#
# # 加载Q表
q_table = np.load("q_table.npy")  # 训练后保存的Q表

# 测试智能体
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    x, y = state
    action = np.argmax(q_table[x, y])
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print(f"动作: {action}, 奖励: {reward}")

env.render()
print(f"总奖励: {total_reward}")
