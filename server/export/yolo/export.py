from ultralytics import YOLO

# 加载 YOLOv8 模型（可替换为 yolov8n.pt / yolov5nu.pt / 自训模型）
model = YOLO("yolov5nu.pt")

# 导出为 TorchScript
model.export(format="torchscript")

# 会生成文件：yolov5nu.torchscript.pt
print("✅ 导出成功：yolov5nu.torchscript")