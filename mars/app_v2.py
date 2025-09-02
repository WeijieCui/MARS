import gradio as gr
import numpy as np
import cv2
from PIL import Image
import time


# ========== 模拟 RL+YOLO 搜索过程 ==========
def rl_search_and_detect(image_pil, steps=10, delay=0.5):
    """
    模拟 RL+YOLO 检测过程，逐步 yield (结果图, heatmap)，实现流式更新。
    """
    if image_pil is None:
        return

    img_bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 初始化 10x10 heatmap
    grid = np.zeros((10, 10), dtype=float)

    # 初始化 ROI
    win_w, win_h = W // 4, H // 4
    cx, cy = W // 2, H // 2

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

        # 画结果图 (带 ROI 红框)
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
        overlay_pil = Image.fromarray(overlay_rgb)
        heatmap_pil = Image.fromarray(heatmap)

        # 每一步 yield 出去
        yield overlay_pil, heatmap_pil

        # 控制速度
        time.sleep(delay)


# ========== Gradio 界面 ==========
with gr.Blocks() as demo:
    gr.Markdown("## 遥感图像目标检测（RL + YOLO 实时流式演示）")

    with gr.Row():
        with gr.Column(scale=2):
            out_image = gr.Image(type="pil", label="检测结果 (实时更新)")
        with gr.Column(scale=1):
            heatmap = gr.Image(type="pil", label="搜索矩阵 Heatmap (实时更新)")

    with gr.Row():
        upload = gr.Image(type="pil", label="上传遥感图", interactive=True, show_label=True)
        steps = gr.Slider(5, 30, value=10, step=1, label="搜索步数")
        btn = gr.Button("开始检测 (流式)")

    btn.click(
        fn=rl_search_and_detect,
        inputs=[upload, steps],
        outputs=[out_image, heatmap],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()
