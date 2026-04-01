from ultralytics import YOLO

# 【核心改动】
# 1. 传入你魔改后的 yaml 配置文件名（如果报错找不到，可以写这个 yaml 的绝对路径）
# 2. 加上 .load("yolov8n.pt") 来加载原版预训练权重
model = YOLO("yolov8-ghost-ema.yaml").load("yolov8n.pt") 

# 训练时直接传入数据增强参数
results = model.train(
    data="/kaggle/working/yolov8-third/My First Project.v1i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
    # 下面是你设置的数据增强参数
    mosaic=1.0,      # 100% 开启马赛克拼接
    hsv_h=0.015,     # 色调变化幅度
    translate=0.1,   # 平移
    scale=0.5,       # 缩放
    fliplr=0.5,      # 50%概率左右翻转
    flipud=0.0       # 0%概率上下翻转
)