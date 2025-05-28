import torch
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms.functional as F
import torchvision.ops as ops
import os
from ultralytics import YOLO 
def draw_boxes_on_image(image_path, detections, output_path="result.jpg"):
    orig_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(orig_img)

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        font = ImageFont.truetype(font_path, size=24)  # 设置更大的字体
    except Exception as e:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        label = det['label']

        # 绘制框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # 文本标签
        text = f"Class:{label} {score:.2f}"
        text_size = draw.textbbox((x1, y1), text, font=font)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]

        # 背景框
        text_bg = [x1, y1 - text_h - 12, x1 + text_w + 12, y1]
        draw.rectangle(text_bg, fill="red")

        # 放置文字
        draw.text((x1 + 2, y1 - text_h - 12), text, fill="white", font=font)

    orig_img.save(output_path)
    print(f"✅ 检测完成，保存至 {output_path}")


# ---------------------------
# 主函数流程
# ---------------------------
IMAGE_PATH = "bus.jpg"
MODEL_PATH = "yolov5nu.pt"
OUTPUT_IMAGE_PATH = "bus_detected_ultralytics.jpg" # Changed output name to avoid overwriting

assert os.path.exists(MODEL_PATH), f"模型文件 {MODEL_PATH} 不存在"
assert os.path.exists(IMAGE_PATH), f"图像 {IMAGE_PATH} 不存在"

print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
model.eval() # Although YOLO class handles this, it's good practice.
print("Model loaded successfully.")

# Inference
print(f"Running inference on {IMAGE_PATH}...")
start = time.time()
results = model.predict(source=IMAGE_PATH, conf=0.3, iou=0.5, save=False, verbose=False)
print(results)
end = time.time()

# Process results and draw
all_detections = []
if results:
    # results[0] because we are processing a single image
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0].cpu().numpy() # Bounding box in xyxy format
        conf = r.conf.item()                     # Confidence score
        cls = int(r.cls.item())                  # Class ID

        all_detections.append({
            'box': [x1, y1, x2, y2],
            'score': conf,
            'label': cls
        })

# Draw the boxes using our helper function
draw_boxes_on_image(IMAGE_PATH, all_detections, output_path=OUTPUT_IMAGE_PATH)


print(f"\n🕒 Inference time: {(end - start) * 1000:.2f} ms")