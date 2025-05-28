import argparse
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pandas as pd
import torch
from torchvision.ops import nms

# COCO_CLASSES 同你的文件读取
with open("coco_classes.txt") as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

def letterbox(im, new_size=(640, 640), color=(114, 114, 114)):
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
import torchvision.transforms.functional as F
def preprocess(image_path, img_size=640):
    im = Image.open(image_path).convert('RGB')
    im_lb, scale, pad_x, pad_y = letterbox(im, new_size=(img_size, img_size))
    img = F.to_tensor(im_lb).unsqueeze(0)  # 1×C×H×W
    return img, scale, pad_x, pad_y, im.size  # 原始大小用于恢复坐标

def postprocess(pred, scale, pad_x, pad_y, orig_size,
                conf_thresh=0.3, iou_thresh=0.5):
    # [1, 84, 8400] → [8400, 84]
    pred = torch.tensor(pred)
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
    # 保留 score 高于 sigma + mean 的
    score_th = scores.std() + scores.mean()
    # 从 coco_classes.txt 中获取类别名称

    for idx in keep:
        if scores[idx] < score_th:
            continue
        detections.append({
            'box': xyxy[idx].tolist(),
            'score': scores[idx].item(),
            'label': COCO_CLASSES[int(cls[idx].item())],
        })
    return detections

def load_font(size=24):
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNSDisplay.ttf", size)
    except:
        return ImageFont.load_default()

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


def main():
    parser = argparse.ArgumentParser(description="YOLO 客户端命令行推理")
    parser.add_argument("--image", type=str, required=True, help="待检测图片路径", default="example/bus.jpg")
    args = parser.parse_args()

    input_np, scale, pad_x, pad_y, orig_size = preprocess(args.image)

    payload = {
        "inputs": {
            "0": {
                "dtype": "float32",
                "shape": [1, 3, 640, 640],
                "data_f32": input_np.flatten().tolist()
            }
        }
    }
    args.model = "http://127.0.0.1:3030/infer"
    print(f"开始请求模型推理服务：{args.model}")
    res = requests.post(args.model, json=payload)

    if res.ok:
        data = res.json()
        if data.get("success"):
            print(f"推理成功，耗时 {data['duration_ms']} ms")
            output_data = np.array(data["outputs"]["196"]["data_f32"], dtype=np.float32)
            output_data = output_data.reshape(1, 84, 8400)
            
            detections = postprocess(output_data, scale, pad_x, pad_y, orig_size)
            draw_boxes_on_image(args.image, detections, "result.jpg")
            print(f"检测结果图像已保存为 result.jpg")

            if detections:
                print("Top 5 检测结果:")
                top5 = sorted(detections, key=lambda x: -x["score"])[:5]
                df = pd.DataFrame([{
                    "类别": d["label"],
                    "置信度": round(d["score"], 3),
                    "位置": f"({int(d['box'][0])}, {int(d['box'][1])}, {int(d['box'][2])}, {int(d['box'][3])})"
                } for d in top5])
                print(df.to_string(index=False))
            else:
                print("未检测到目标或置信度过低")
        else:
            print("推理失败:", data.get("message", "未知错误"))
    else:
        print("请求失败，HTTP 状态码：", res.status_code)
        print(res.text)

if __name__ == "__main__":
    main()