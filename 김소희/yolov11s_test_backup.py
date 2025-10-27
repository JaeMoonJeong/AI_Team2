from ultralytics import YOLO
import torch

# GPU 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

# 데이터 경로
data_yaml = "C:/Users/wockd/OneDrive/바탕 화면/말하는 감자/스프린트/혼자테스트/data/dataset_yolo/data.yaml"

# 이 부분 추가!
if __name__ == '__main__':
    # YOLOv11s 모델 로드
    model = YOLO("yolo11s.pt")

    # 학습 시작 (workers=0 추가)
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs_yolo11s",
        name="train",
        device=device,
        workers=0  # 이 줄 추가 (멀티프로세싱 비활성화)
    )

    print("학습 완료!")
    print(f"결과 저장 위치: runs_yolo11s/train")