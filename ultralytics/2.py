from ultralytics import YOLO

# 1. 加载你刚刚训练好的“魔改版”最优权重
# 注意：路径要换成你实际训练生成的 best.pt 的绝对路径
# 在 Kaggle 中，通常会保存在类似下面这个路径：
weight_path = "/kaggle/working/yolov8-third/runs/detect/train2/weights/best.pt" 
model = YOLO(weight_path)

# 2. 导出为 ONNX 格式
export_path = model.export(
    format="onnx",      # 指定目标格式
    imgsz=640,          # 输入图片尺寸，必须与你训练和实际推理时保持一致
    half=False,         # FP16 半精度导出。如果你的部署显卡/设备支持 FP16，设为 True 能大幅提速
    dynamic=False,      # 动态输入尺寸。为了追求极限 FPS，通常设为 False（固定尺寸推理最快）
    simplify=True,      # 💡 强烈建议开启！调用 onnxsim 自动精简和优化计算图，去掉冗余算子
    opset=12            # ONNX 算子集版本，12 或 11 兼容性最好，通常保持默认即可
)

print(f"ONNX 模型已成功导出，保存在: {export_path}")