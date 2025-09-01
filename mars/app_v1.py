# -*- coding: utf-8 -*-
"""
Gradio 交互 + RL 搜索 + OBB 可视化（后端占位实现）
- 10x10 探索矩阵: 未访问=0, 未检出=-1, 命中=置信度
- 动作空间: up/down/left/right/zoom_in/zoom_out
- 检测器接口可替换为真实 YOLO (yolo11)
"""
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import math
from typing import List, Dict, Any

from mars.agent import RLAgent
from mars.detector import RealYoloDetector, BaseDetector
from mars.env import SearchEnv


# ----------------------------
# 工具函数：OBB 和绘制
# ----------------------------
def angle_to_box(cx, cy, w, h, theta_deg):
    """由中心点、宽高、角度(度)生成 OBB 的四个顶点 (int)"""
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy],
    ], dtype=float)
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    rot = corners @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot.astype(int)


def draw_obbs(img_bgr: np.ndarray, obbs: List[Dict[str, Any]], color=(0, 255, 0), thickness=3):
    """在 BGR 图上绘制一组 OBB（四点连线）"""
    out = img_bgr.copy()
    for obb in obbs:
        pts = angle_to_box(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["theta"])
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def to_heatmap_image(grid: np.ndarray) -> np.ndarray:
    """
    将 10x10 网格渲染为热力图：
    -1 -> 蓝色（未检出）
     0 -> 灰色（未访问）
    >0 -> 从黄到红（置信度越高越红）
    """
    assert grid.shape == (10, 10)
    norm = np.zeros_like(grid, dtype=float)
    visited_mask = (grid != 0)
    positives = np.maximum(grid, 0)
    if np.any(positives > 0):
        vmax = positives.max()
        if vmax > 0:
            norm = positives / vmax
    # 生成彩色图
    H, W = 10, 10
    cell = 32  # 可视化每格大小
    vis = np.zeros((H * cell, W * cell, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            val = grid[i, j]
            c0, c1, c2 = 160, 160, 160  # 未访问：灰
            if val == -1:
                c0, c1, c2 = 180, 120, 0  # BGR：偏蓝（可自行调整）
            elif val > 0:
                # 由黄(0,255,255) -> 红(0,0,255) 的简易渐变（HSV 更优，这里简化）
                t = norm[i, j]
                c0, c1, c2 = 0, int(255 * (1 - t)), 255
            vis[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = (c0, c1, c2)
            # 画网格线
            cv2.rectangle(vis, (j * cell, i * cell), ((j + 1) * cell - 1, (i + 1) * cell - 1), (40, 40, 40), 1)
    return vis


# ----------------------------
# 管线：一次按钮点击内执行若干步搜索
# ----------------------------
def rl_search_and_detect(image_pil: Image.Image, target_type: str, steps: int = 12, use_real_yolo: bool = False):
    """
    对输入图像运行 RL 搜索 + 检测若干步，输出：
    - 叠加 OBB 的图像
    - 探索热力图（10x10）
    """
    if image_pil is None:
        return None, None

    img_bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 选择检测器
    if use_real_yolo:
        detector = RealYoloDetector(target_type=target_type)
    else:
        detector = BaseDetector(target_type=target_type)

    env = SearchEnv(img_bgr, detector, grid_n=10)
    agent = RLAgent(grid_shape=(10, 10), eps=0.15, alpha=0.5, gamma=0.9)

    all_obbs: List[Dict[str, Any]] = []

    # RL 回合
    i, j = env._cell_of(env.cx, env.cy)
    s = (i, j, env.scale_idx)
    last_action = None
    for t in range(steps):
        a = agent.select_action(*s)
        s_old = s
        s, r, obbs, s2 = env.step(a)
        agent.update(s_old, a, r, s2)
        all_obbs.extend(obbs)
        last_action = a
        s = s2

    # 绘制输出
    overlay = draw_obbs(img_bgr, all_obbs, color=(0, 255, 0), thickness=3)
    heatmap = to_heatmap_image(env.grid)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb)


# ----------------------------
# Gradio 界面（前后端打通）
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Mars Detector(RL + OBB)")

    with gr.Row():
        with gr.Column(scale=2):
            out_image = gr.Image(type="pil", label="Result")
            out_grid = gr.Image(type="pil", label="Exploring Image")
        with gr.Column(scale=1):
            in_image = gr.Image(type="pil", label="Upload an Image", interactive=True)
            target = gr.Dropdown(["Plane", "Ship", "Car", "Basket Ball Pitch"], label="Target", value="Plane")
            steps = gr.Slider(4, 40, value=12, step=1, label="Max Steps")
            use_real = gr.Checkbox(False, label="YOLO V11")
            btn = gr.Button("Detect")

    btn.click(
        fn=rl_search_and_detect,
        inputs=[in_image, target, steps, use_real],
        outputs=[out_image, out_grid]
    )

if __name__ == "__main__":
    demo.launch()
