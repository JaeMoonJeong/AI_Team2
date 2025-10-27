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
    # YOLOv11m 학습
    print("\n=== YOLOv11m 학습 시작 ===")
    model_11m = YOLO("yolo11m.pt")
    results_11m = model_11m.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        project="runs/train",
        name="yolo11m"
    )
    
    print("\n=== 모든 학습 완료 ===")