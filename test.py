from ultralytics import YOLO

# 多进程保护（Windows必须）
if __name__ == '__main__':
    # 加载训练好的最佳权重（和train.py的name对应）
    model = YOLO("D:/yolo11/ultralytics-main/runs/obb1/qjd_classes_obb6/weights/best.pt")    
    # 对测试集执行验证，计算P/R/mAP50等指标（和命令行val一致）
    test_results = model.val(
        data="D:/yolo11/ultralytics-main/ultralytics/cfg/datasets/smoke_obb.yaml",    
        imgsz=640,     # 和训练一致
        device="0",    # 和训练一致
        batch=16,       # 和训练一致
    )
