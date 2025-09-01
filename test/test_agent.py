import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import gym
import numpy as np
from grid_world import GridWorldEnv

# 注册带渲染的环境
gym.register(
    id='GridWorld-Render-v1',
    entry_point=GridWorldEnv,
    kwargs={'render_mode': 'human'}  # 使用 'human' 模式
)

# 创建环境
env = gym.make('GridWorld-Render-v1')

# 加载保存的Q表
try:
    q_table = np.load("q_table.npy")
    print("Q表已加载")
except FileNotFoundError:
    print("警告：找不到Q表，使用随机策略")
    q_table = np.zeros((4, 4, 4))

# 测试智能体
state, _ = env.reset()
terminated = False
truncated = False
total_reward = 0
step_count = 0
max_steps = 50  # 防止无限循环

print("初始状态:")
env.render()  # 初始渲染

while not (terminated or truncated) and step_count < max_steps:
    x, y = state
    # 确保坐标在Q表范围内
    x = np.clip(x, 0, 3)
    y = np.clip(y, 0, 3)

    action = np.argmax(q_table[x, y])
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    step_count += 1

    print(f"步数 {step_count}: 位置 ({x}, {y}), 动作 {['上', '右', '下', '左'][action]}, 奖励 {reward:.1f}")
    state = next_state

print("\n最终状态:")
env.render()
print(f"位置: {state}")
print(f"总奖励: {total_reward:.1f}")
print(f"总步数: {step_count}")
env.close()