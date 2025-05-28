import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.ops import nms
import numpy as np
import os

# ----------------------------
# 1. Letterbox 缩放函数
# ----------------------------
def letterbox(im, new_size=(640, 640), color=(114,114,114)):
    orig_w, orig_h = im.size
    new_w, new_h = new_size
    scale = min(new_w / orig_w, new_h / orig_h)
    resize_w, resize_h = int(orig_w * scale), int(orig_h * scale)
    im_resized = im.resize((resize_w, resize_h), Image.BILINEAR)
    canvas = Image.new('RGB', new_size, color)
    pad_x = (new_w - resize_w) // 2
    pad_y = (new_h - resize_h) // 2
    canvas.paste(im_resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y

# ----------------------------
# 2. 图像预处理
# ----------------------------
def preprocess(image_path, img_size=640):
    im = Image.open(image_path).convert('RGB')
    im_lb, scale, pad_x, pad_y = letterbox(im, new_size=(img_size, img_size))
    img = F.to_tensor(im_lb).unsqueeze(0)  # 1×C×H×W
    return img, scale, pad_x, pad_y, im.size  # 原始大小用于恢复坐标

# ----------------------------
# 3. 后处理 & NMS
# ----------------------------
import torch
from torchvision.ops import nms

def postprocess(pred, scale, pad_x, pad_y, orig_size,
                conf_thresh=0.3, iou_thresh=0.5):
    # [1, 84, 8400] → [8400, 84]
    pred = pred.squeeze(0).permute(1, 0).contiguous()  # shape: (8400, 84)
    boxes = pred[:, :4]
    class_scores = pred[:, 4:]

    # 取最大类别得分和其 index
    conf, cls = class_scores.max(1)  # [8400]
    scores = conf  # 这里就是最终用于 NMS 的置信度
    keep_mask = scores > conf_thresh

    if keep_mask.sum() == 0:
        return []

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    cls = cls[keep_mask]

    # xywh → xyxy
    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # 缩放回原图尺寸
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

    orig_w, orig_h = orig_size
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, orig_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, orig_h)

    keep = nms(xyxy, scores, iou_thresh)

    detections = []
    for idx in keep:
        detections.append({
            'box': xyxy[idx].tolist(),
            'score': scores[idx].item(),
            'label': cls[idx].item()
        })
    return detections

# ----------------------------
# 4. 绘图函数
# ----------------------------
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

# ----------------------------
# 5. 主流程
# ----------------------------
if __name__ == "__main__":
    IMAGE_PATH = "bus.jpg"
    MODEL_PATH = "yolov5nu.torchscript"
    OUTPUT_PATH = "bus_detected_torchscript.jpg"

    assert os.path.exists(MODEL_PATH), f"模型文件 {MODEL_PATH} 不存在"
    assert os.path.exists(IMAGE_PATH), f"图像 {IMAGE_PATH} 不存在"

    model_ts = torch.jit.load(MODEL_PATH, map_location="cpu")
    model_ts.eval()

    img_tensor, scale, pad_x, pad_y, orig_size = preprocess(IMAGE_PATH)
    with torch.no_grad():
        pred = model_ts(img_tensor)

    dets = postprocess(pred, scale, pad_x, pad_y, orig_size)
    print(f"检测结果：{dets}")
    draw_boxes_on_image(IMAGE_PATH, dets, output_path=OUTPUT_PATH)