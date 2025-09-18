# -*- coding: utf-8 -*-
"""
MARS Application
- Action Space: up/down/left/right/zoom_in/zoom_out
"""
import os

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import math
from typing import List, Dict, Any

from rl_model import RLQtableModel, MODEL_DIR
from detector import YoloV11Detector, BaseDetector
from env import SearchEnv

custom_css = """
.custom-checkbox {
    # padding-top: 26px;
}
.custom-message {
    font-size: 10px;
}
"""

custom_css_flex = """
.flex-row {
    flex-direction: row;
}
"""
visual_model_dict = {}
qtable_model_dict = {}
env = SearchEnv()
should_continue = True
AGENT_VERSIONS = os.listdir(MODEL_DIR)
AGENT_VERSIONS.append("")
agent_versions = AGENT_VERSIONS


def angle_to_box(cx, cy, w, h, theta_deg):
    """
    Calculate angle to box
    :param cx: center x
    :param cy: center y
    :param w: width
    :param h: height
    :param theta_deg: orient angle 
    :return: 
    """
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


box_color_dict = {
    0: (255, 0, 0),
    1: (255, 125, 0),
    2: (255, 255, 0),
    3: (255, 0, 125),
    4: (255, 0, 255),
    5: (0, 255, 0),
    6: (0, 255, 125),
    7: (0, 255, 255),
    8: (0, 0, 125),
    9: (0, 0, 255),
    10: (125, 0, 0),
    11: (125, 125, 0),
    12: (125, 255, 0),
    13: (125, 0, 125),
    14: (125, 0, 255),
}


def draw_obbs(img_bgr: np.ndarray,
              obbs: List[Dict[str, Any]],
              window: [],
              thickness=3):
    """
    draw orient bounding boxes
    :param img_bgr: image arrays
    :param obbs: orient bounding boxes
    :param window: window
    :param thickness: thickness of box bound
    :return:
    """
    out = img_bgr.copy()
    for obb in obbs:
        pts = angle_to_box(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["theta"])
        cv2.polylines(out, [pts], isClosed=True, color=box_color_dict[obb['class']], thickness=thickness)
    if window is not None:
        cv2.polylines(out, [window], isClosed=True, color=(255, 0, 0), thickness=thickness)
    return out


def to_heatmap_image(grid: np.ndarray) -> np.ndarray:
    """
    10x10 Heat Map Image
    -1 -> Blue (No found)
     0 -> Grey (Unknown)
    >0 -> Yellow to Red, confident
    """
    norm = np.zeros_like(grid, dtype=float)
    visited_mask = (grid != 0)
    positives = np.maximum(grid, 0)
    if np.any(positives > 0):
        vmax = positives.max()
        if vmax > 0:
            norm = positives / vmax
    # Draw Grid
    H, W = grid.shape
    cell = 320 // H  # Grid size
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


def get_detector(model: str, device: str):
    if model in visual_model_dict:
        return visual_model_dict.get(model)
    if model == 'YOLO_V11':
        detector = YoloV11Detector(device=device)
    else:
        detector = BaseDetector()
    visual_model_dict.setdefault(model, detector)
    return detector


def get_agent(model: str, agent_version: str, save: bool = False, load=True, image_url: str = "default"):
    if model in qtable_model_dict:
        return qtable_model_dict.get(model)
    if model == 'QTable':
        filename = os.path.basename(image_url)
        if agent_version:
            rl_model = RLQtableModel(save=save, load=load, model=agent_version)
        else:
            rl_model = RLQtableModel(save=save, load=load, model='{}_qtable.pkl'.format(filename))
        visual_model_dict.setdefault(image_url, rl_model)
    else:
        rl_model = BaseDetector()
    return rl_model


def update_file(
        image_url: str,
):
    """
    uploading a File: Update
    """
    if image_url is None:
        return None, None, None, None, gr.Button("Detect", interactive=False)
    img_bgr = cv2.imread(image_url)
    h, w = img_bgr.shape[:2]
    max_size = max(w, h)
    limit = 3000
    if max_size > limit:
        if w >= h:
            w, h = limit, int(h / w * limit)
        else:
            w, h = int(w / h * limit), limit
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    all_obbs: List[Dict[str, Any]] = []
    env.set_image(img_bgr)
    # Update outputs
    overlay = draw_obbs(img_bgr, all_obbs, window=env.border(), thickness=3)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(to_heatmap_image(env.get_grid()), cv2.COLOR_BGR2RGB)
    view = cv2.cvtColor(env.view(), cv2.COLOR_BGR2RGB)
    return (Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(view), "",
            gr.Button("Detect", interactive=True))


def rl_search_and_detect(
        image_url: str,
        target: str,
        steps: str = '12',
        visual_model: str = 'YOLO_V11',
        agent_name: str = 'QTable',
        agent_version: str = '',
        device: str = 'cpu',
        save=False,
        load=False,
):
    """
    Search and detect objects
    :param image_url: image url
    :param target: target, default: all
    :param steps: steps
    :param visual_model: model name
    :param agent_name: agent name
    :param save: save model
    :param load: load model
    :return:
    """
    if image_url is None:
        return None, None, None, "No Image"
    global should_continue
    should_continue = True
    img_bgr = cv2.imread(image_url)
    # Env
    detector = get_detector(visual_model, device)
    env.set_image(img_bgr)
    env.set_detector(detector)
    env.set_target(target)
    # RL iterations
    agent = get_agent(agent_name, agent_version, save=save, load=load, image_url=image_url)
    status, reward, obbs, new_obbs, window = env.reset()
    # Wrap outputs
    heatmap = to_heatmap_image(env.get_grid())
    overlay = draw_obbs(img_bgr, env.found_objects, window=env.border(), thickness=3)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    yield Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(
        window), "Steps: {} / {}, found: {}. {}".format(
        0, steps, len(obbs), 'Done.' if len(status[3]) == 0 else '')
    overlay_rgb = None
    heatmap_rgb = None
    for t in range(int(steps)):
        action = agent.select_action(*status)
        if not action:
            yield (Image.fromarray(overlay_rgb) if overlay_rgb is not None else None,
                   Image.fromarray(heatmap_rgb) if heatmap_rgb is not None else None,
                   Image.fromarray(window), "Found: {}. Done.".format(len(obbs)))
            break
        status_new, reward, obbs, new_obbs, window = env.step(action)
        agent.update(status, action, reward, status_new)
        status = status_new
        heatmap = to_heatmap_image(env.get_grid())
        # Wrap outputs
        overlay = draw_obbs(img_bgr, env.found_objects, window=env.border(), thickness=3)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb), Image.fromarray(
            window), "Steps: {} / {}, found: {}. {}".format(
            t + 1, steps, len(obbs), 'Done.' if len(status_new[3]) == 0 else '')
        if not should_continue or len(status_new[3]) == 0:
            break
    if save:
        filename = os.path.basename(image_url)
        agent.save("{}_qtable.pkl".format(filename))


def interrupt_function():
    """
    interrupt searching process.
    :return: message
    """
    global should_continue
    should_continue = False
    return "Interrupted."


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## MARS")
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
                    ],
                        label="Target",
                        value="-1")
                    steps_dropdown = gr.Dropdown(["0", "5", "10", "20", "50", "100", "150", "200"], label="Max Steps",
                                                 value="10")
                    device_dropdown = gr.Dropdown(["cpu", "gpu"], label="Device", value="cpu")
                    visual_model = gr.Dropdown(["YOLO_V11"], label="Visual Model", value="YOLO_V11")
                    agent_version = gr.Dropdown(agent_versions, label="RL Version", value=''
                    if 'qtable.pkl' in agent_versions else agent_versions[0])
                    agent_model = gr.Dropdown(["QTable", "NN"], label="RL Model", value="QTable")
                    save_radio = gr.Checkbox(False, label="SaveRLModel", elem_classes="custom-checkbox")
                    load_model_radio = gr.Checkbox(True, label="LoadRLModel", elem_classes="custom-checkbox")
                    in_image = gr.File(label="Upload an Image", height=30)
                with gr.Row():
                    btn = gr.Button("Detect", interactive=False)
                    stop_btn = gr.Button("Stop")
                message = gr.Label(label="Progress:", value="", elem_classes="custom-message")
    in_image.change(
        fn=update_file,
        inputs=[in_image],
        outputs=[out_image, heatmap, window, message, btn]
    )
    btn.click(
        fn=rl_search_and_detect,
        inputs=[in_image, target_dropdown, steps_dropdown, visual_model, agent_model, agent_version, device_dropdown,
                save_radio,
                load_model_radio],
        outputs=[out_image, heatmap, window, message],
    )
    stop_btn.click(
        fn=interrupt_function,
        inputs=[],
        outputs=[message]
    )

if __name__ == "__main__":
    demo.launch()
