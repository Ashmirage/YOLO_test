from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt") # 加载模型
    model.train(
        data = r"african-wildlife.yaml", #选择coco数据集
        epochs = 10, # 训练轮数
        imgsz = 640, # 必须要是32的倍数, 图片的变换: 先把最长边缩放成设置的系数,然后短边再等比例缩放, 之后将短边通过padding, 整体变成一个方阵, imgsz影响训练效率,但是不是越小越好
        # 越小,图片的细节丢失越多,数据变差, 太大,一方面增加训练时间,一方面如果目标本来就很小,反而降低质量
        batch = 2,
        cache = False,
        workers = 0,
    )


