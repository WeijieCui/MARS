import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import gym
import numpy as np
from grid_world import GridWorldEnv

# 注册环境
gym.register(
    id='GridWorld-v1',
    entry_point=GridWorldEnv,
    kwargs={'render_mode': None}
)

# 创建环境
env = gym.make('GridWorld-v1')

# Q-learning 参数
q_table = np.zeros((4, 4, 4))  # [x, y, action]
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
episodes = 1000

for episode in range(episodes):
    # 使用新API的reset方式
    state, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # ε-贪婪策略
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x, y = state
            action = np.argmax(q_table[x, y])

        # 执行动作 - 使用新API返回5个值
        next_state, reward, terminated, truncated, _ = env.step(action)

        # 更新Q值
        x, y = state
        next_x, next_y = next_state

        # 确保索引在范围内
        next_x = np.clip(next_x, 0, 3)
        next_y = np.clip(next_y, 0, 3)

        q_table[x, y, action] += learning_rate * (
                reward +
                discount_factor * np.max(q_table[next_x, next_y]) -
                q_table[x, y, action]
        )

        state = next_state

# 保存训练好的Q表
np.save("q_table.npy", q_table)
print(f"训练完成！Q表已保存，共训练 {episodes} 回合")