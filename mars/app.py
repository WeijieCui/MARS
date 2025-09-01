import gradio as gr
import numpy as np
import cv2


# 模拟目标检测函数 (你可以替换为真实模型推理)
def detect_objects(image, target_type):
    if image is None:
        return None

    # 将gradio image (PIL.Image) 转 numpy
    img = np.array(image.convert("RGB"))

    # 假装检测到一个 oriented bounding box
    h, w, _ = img.shape
    box = np.array([
        [w // 3, h // 3],
        [w // 2, h // 3],
        [w // 2 + 20, h // 2],
        [w // 3 - 20, h // 2]
    ], dtype=np.int32)

    overlay = img.copy()
    cv2.polylines(overlay, [box], isClosed=True, color=(255, 0, 0), thickness=3)

    return overlay


with gr.Blocks() as demo:
    gr.Markdown("## Mars Model")

    with gr.Row():
        with gr.Column(scale=2):
            image_display = gr.Image(type="pil", label="Image Show")
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload a picture", interactive=True)
            target_type = gr.Dropdown(
                ["Plane", "Ship", "Car"],
                label="Target"
            )
            detect_btn = gr.Button("Detect")

    # 输出显示检测结果
    detect_btn.click(
        fn=detect_objects,
        inputs=[input_image, target_type],
        outputs=image_display
    )

if __name__ == "__main__":
    demo.launch(share=False)
