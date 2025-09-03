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

from mars.agent import RLQtableAgent
from mars.detector import YoloV11Detector, BaseDetector
from mars.env import SearchEnv
from mars.utils import merge_bounding_box

custom_css = """
.custom-checkbox {
    padding-top: 26px;
}
"""

custom_css_flex = """
.flex-row {
    flex-direction: row;
}
"""
visual_model_dict = {}
agent_model_dict = {}
env = SearchEnv()


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
    10x10 Heat Map Image
    -1 -> Blue (No found)
     0 -> Grey (Unknown)
    >0 -> Yellow to Red, confident
    """
    assert grid.shape == (10, 10)
    norm = np.zeros_like(grid, dtype=float)
    visited_mask = (grid != 0)
    positives = np.maximum(grid, 0)
    if np.any(positives > 0):
        vmax = positives.max()
        if vmax > 0:
            norm = positives / vmax
    # Draw Grid
    H, W = 10, 10
    cell = 32  # Grid size
    vis = np.zeros((H * cell, W * cell, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            val = grid[i, j]
            c0, c1, c2 = 160, 160, 160  # Unknown
            if val == -1:
                c0, c1, c2 = 180, 120, 0  # BGR: Blue
            elif val > 0:
                # Yellow(0,255,255) -> Red(0,0,255)
                t = norm[i, j]
                c0, c1, c2 = 0, int(255 * (1 - t)), 255
            vis[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = (c0, c1, c2)
            # Draw grid lines
            cv2.rectangle(vis, (j * cell, i * cell), ((j + 1) * cell - 1, (i + 1) * cell - 1), (40, 40, 40), 1)
    return vis


def get_detector(model: str):
    if model in visual_model_dict:
        return visual_model_dict.get(model)
    if model == 'YOLO_V11':
        detector = YoloV11Detector()
    else:
        detector = BaseDetector()
    visual_model_dict.setdefault(model, detector)
    return detector


def get_agent(model: str, training: bool = False, load=True):
    if model in agent_model_dict:
        return agent_model_dict.get(model)
    if model == 'QTable':
        agent = RLQtableAgent(training=training, load=load, model='qtable-7.pkl')
    else:
        agent = BaseDetector()
    visual_model_dict.setdefault(model, agent)
    return agent


# ----------------------------
# Uploading a File: Update
# ----------------------------
def update_file(
        image_url: str,
):
    """
    Update file
    """
    if image_url is None:
        return None, None, None, None, gr.Button("Detect", interactive=False)
    img_bgr = cv2.imread(image_url)
    all_obbs: List[Dict[str, Any]] = []
    # 绘制输出
    overlay = draw_obbs(img_bgr, all_obbs, color=(0, 255, 0), thickness=3)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    env.set_image(img_bgr)
    heatmap_rgb = cv2.cvtColor(to_heatmap_image(env.grid), cv2.COLOR_BGR2RGB)
    view = cv2.cvtColor(env.view(), cv2.COLOR_BGR2RGB)
    return (Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(view), "",
            gr.Button("Detect", interactive=True))


# ----------------------------
# 管线：一次按钮点击内执行若干步搜索
# ----------------------------
def rl_search_and_detect(
        image_url: str,
        target: str,
        steps: int = 12,
        visual_model: str = 'YOLO_V11',
        agent_name: str = 'QTable',
        training=False,
):
    """
    对输入图像运行 RL 搜索 + 检测若干步，输出：
    - 叠加 OBB 的图像
    - 探索热力图（10x10）
    """
    if image_url is None:
        return None, None, None, None
    img_bgr = cv2.imread(image_url)
    # Env
    detector = get_detector(visual_model)
    env.set_image(img_bgr)
    env.set_detector(detector)
    env.set_target(target)
    # RL iterations
    agent = get_agent(agent_name, training=training)
    all_obbs: List[Dict[str, Any]] = []
    status = env.reset()
    for t in range(int(steps)):
        action = agent.select_action(*status)
        if not action:
            break
        status_new, reward, obbs, window = env.step(action)
        agent.update(status, action, reward, status_new)
        merge_bounding_box(all_obbs, obbs)
        status = status_new
        heatmap = to_heatmap_image(env.grid)

        # Wrap outputs
        overlay = draw_obbs(img_bgr, all_obbs, color=(0, 255, 0), thickness=3)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(
            window), "Detecting: {} / {}".format(t + 1, steps)
    if training:
        agent.save()
    return Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(window), "Done"


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Mars Detector(RL + OBB)")

    with gr.Row():
        with gr.Column(scale=2):
            out_image = gr.Image(type="pil", label="Result", height=650, width=760)
        with gr.Column(scale=1):
            with gr.Blocks(css="custom_css_flex"):
                with gr.Row():
                    window = gr.Image(type="pil", label="Exploring Image", width=200, height=220, interactive=False)
                    heatmap = gr.Image(type="pil", label="Exploring Image", width=200, height=220)
            with gr.Column():
                with gr.Row():
                    target_dropdown = gr.Dropdown([
                        ('All', '-1'),
                        ('Plane', '0'),
                        ('Ship', '1'),
                        ('Storage tank', '2'),
                        ('Baseball Diamond', '3'),
                        ('Tennis Court', '4'),
                        ('Basketball Court', '5'),
                        ('Ground Track Field', '6'),
                        ('Harbor', '7'),
                        ('Bridge', '8'),
                        ('Larget Vehicle', '9'),
                        ('Small Vehicle', '10'),
                        ('Helicopter', '11'),
                        ('Roundabout', '12'),
                        ('Soccer Ball Field', '13'),
                        ('Swimming Poll', '14'),
                        # ({0: 'plane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond', 4: 'tennis court',
                        #   5: 'basketball court', 6: 'ground track field', 7: 'harbor', 8: 'bridge', 9: 'large vehicle',
                        #   10: 'small vehicle', 11: 'helicopter', 12: 'roundabout', 13: 'soccer ball field', 14: 'swimming pool'}),
                    ],
                        label="Target",
                        value="-1")
                    steps_dropdown = gr.Dropdown(["5", "10", "20", "50", "100", "150", "200"], label="Max Steps",
                                                 value="10")
                    visual_model = gr.Dropdown(["YOLO_V11"], label="Visual Model", value="YOLO_V11")
                    agent_model = gr.Dropdown(["QTable", "NN"], label="Agent Model", value="QTable")
                    training_radio = gr.Checkbox(False, label="Training", elem_classes="custom-checkbox")
                in_image = gr.File(label="Upload an Image", height=30)
                btn = gr.Button("Detect", interactive=False)
                message = gr.Label(label="Progress: ", value="")
    in_image.change(
        fn=update_file,
        inputs=[in_image],
        outputs=[out_image, heatmap, window, message, btn]
    )
    btn.click(
        fn=rl_search_and_detect,
        inputs=[in_image, target_dropdown, steps_dropdown, visual_model, agent_model, training_radio],
        outputs=[out_image, heatmap, window, message],
    )

if __name__ == "__main__":
    demo.launch()
