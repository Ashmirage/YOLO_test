from ultralytics import YOLO

model = YOLO(r"yolo11l.pt")
print(model.task) # 打印模型的任务
print(model.names) # 答应预测的类型
print(sum(p.numel() for p in model.parameters()))