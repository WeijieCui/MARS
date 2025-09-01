import gradio as gr
import numpy as np
import cv2
from PIL import Image
import time


# ========== 模拟 RL+YOLO 搜索 ==========
def dummy_rl_yolo(image_pil, steps=10):
    """
    模拟 RL+YOLO 检测过程，逐步更新左图(结果+ROI)和右下角 heatmap。
    """
    if image_pil is None:
        return

    img_bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 初始化 10x10 heatmap
    grid = np.zeros((10, 10), dtype=float)

    # 模拟 ROI 移动
    win_w, win_h = W // 4, H // 4
    cx, cy = W // 2, H // 2

    updates = []
    for t in range(steps):
        # 随机移动 ROI
        cx = np.clip(cx + np.random.randint(-40, 41), win_w // 2, W - win_w // 2)
        cy = np.clip(cy + np.random.randint(-40, 41), win_h // 2, H - win_h // 2)
        x1, y1 = cx - win_w // 2, cy - win_h // 2
        x2, y2 = x1 + win_w, y1 + win_h

        # 在 heatmap 上更新对应格子
        i = int(cy / H * 10)
        j = int(cx / W * 10)
        grid[i, j] = max(grid[i, j], np.random.rand())  # 模拟置信度

        # 画结果图
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 当前 ROI 红框
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # 画 heatmap
        heatmap = np.zeros((10, 10, 3), dtype=np.uint8)
        vmax = grid.max() if grid.max() > 0 else 1
        for ii in range(10):
            for jj in range(10):
                val = grid[ii, jj] / vmax
                color = (0, int(255 * (1 - val)), int(255 * val))  # 蓝->红渐变
                heatmap[ii, jj] = color
        heatmap = cv2.resize(heatmap, (200, 200), interpolation=cv2.INTER_NEAREST)

        # 转 PIL
        updates.append((Image.fromarray(overlay_rgb), Image.fromarray(heatmap)))

    return updates


def run_search(image_pil, steps):
    updates = dummy_rl_yolo(image_pil, steps)
    # 只返回最后一次结果
    if updates:
        return updates[-1]
    return None, None


# ========== Gradio 界面 ==========
with gr.Blocks() as demo:
    gr.Markdown("## 遥感图像目标检测（RL + YOLO）")

    with gr.Row():
        with gr.Column(scale=2):
            out_image = gr.Image(type="pil", label="检测结果", show_label=True, height=600, width=800)
        with gr.Column(scale=1):
            heatmap = gr.Image(type="pil", label="搜索矩阵 Heatmap", show_label=True)
            upload = gr.Image(type="pil", label="上传遥感图", interactive=True, visible=True, show_label=True,
                              render=True, mirror_webcam=False, height=240, width=400)
            steps = gr.Slider(5, 30, value=10, step=1, label="搜索步数")
            btn = gr.Button("开始检测")

    btn.click(
        fn=run_search,
        inputs=[upload, steps],
        outputs=[out_image, heatmap]
    )

if __name__ == "__main__":
    demo.launch()
