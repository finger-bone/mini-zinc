import streamlit as st
import requests
from PIL import Image
import numpy as np
import os

# 新增：加载ImageNet标签文件
CLASS_INDEX = {}
with open("imagenet_classes.txt") as f:
    for idx, line in enumerate(f):
        CLASS_INDEX[idx] = line.split(",")[-1].strip()

# 新增：示例图片选择
example_images = ["goldfish.JPEG", "great_white_shark.JPEG", "hammerhead.JPEG", "tiger_shark.JPEG"]
selected_example = st.sidebar.selectbox("选择预设图片", [""] + example_images)

# 页面标题和说明
st.title("ResNet18 推理客户端")
st.write("上传或选择图片进行分类预测")

# 图片加载逻辑合并
if selected_example:
    image_path = os.path.join("./example", selected_example)
    image = Image.open(image_path)
    st.image(image, caption='选择的示例图片', use_container_width=True)
elif uploaded_file := st.file_uploader("上传图片 (支持JPG/PNG)", type=["jpg", "png"]):
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_container_width=True)
else:
    image = None

if image:
    # 预处理流程保持不变
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis].astype(np.float32)
    
    # 新增：添加模型输出处理
    request_data = {
        "inputs": {
            "0": {
                "dtype": "float32",
                "shape": tensor.shape,
                "data_f32": tensor.flatten().tolist()
            }
        }
    }

    # 发送推理请求
    response = requests.post("http://127.0.0.1:3030/infer", json=request_data)
    
    # 解析响应
    result = response.json()
    if result["success"]:
        # 显示推理结果表格
        st.subheader("推理结果")
        outputs = result["outputs"]
        if outputs:
            output_data = np.array(outputs["49"]["data_f32"], dtype=np.float32)
            probabilities = np.exp(output_data) / np.sum(np.exp(output_data))  # 手动计算softmax
            
            # 新增：显示所有概率和标签
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            top5_probs = probabilities[top5_indices]
            top5_labels = [CLASS_INDEX[i] for i in top5_indices]
            
            st.write("前5预测结果：")
            st.table({
                "Label": top5_labels,
                "Probability": f"{np.round(top5_probs * 100, 2)}%"
            })
        
        # 新增推理耗时显示
        st.write(f"推理耗时: {result['duration_ms']} ms ⏱️")
    else:
        st.error(f"推理失败: {result['message']}")
