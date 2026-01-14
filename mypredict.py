from ultralytics import YOLO

# 预测脚本

model = YOLO(r"yolo11n.pt")

model.predict(
    source=r"ultralytics/assets",
    save=True,
    show=False,
    visualize = True,
)