# import 부분
import torch
from ultralytics import YOLO

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 경로
DATA_YAML = r"C:\Users\wockd\OneDrive\바탕 화면\말하는 감자\스프린트\초급 프로젝트\ai05-level1-project\DATASET_YOLO\data.yaml"

# 학습 설정
EPOCHS = 50
IMG_SIZE = 640

if __name__ == '__main__':
    # YOLOv8m 학습
    print("\n=== YOLOv8m 학습 시작 ===")
    model_8m = YOLO("yolov8m.pt")
    results_8m = model_8m.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        project="runs/train",
        name="yolov8m"
    )
    
    print("\n=== 학습 완료 ===")