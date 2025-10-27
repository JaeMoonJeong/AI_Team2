from ultralytics import YOLO
import os

def main():
    # 1. 모델 로드
    # 'yolov8s.pt'는 작고 빠른 모델입니다. (n, m, l, x 등으로 변경 가능)
    # 처음 학습 시에는 사전 학습된 가중치를 불러옵니다.
    model = YOLO('yolov10s.pt') 

    # 2. 데이터셋 YAML 파일 경로
    # 당신이 만든 data.yaml 파일의 절대 경로 또는 상대 경로를 지정합니다.
    data_yaml_path = 'data.yaml'

    # 3. 모델 학습
    # epochs: 전체 데이터셋을 몇 번 반복 학습할지 (예: 100)
    # imgsz: 학습 시 이미지 크기 (예: 640)
    # batch: 한 번에 몇 개의 이미지를 학습할지 (GPU 메모리에 따라 조절)
    print("Starting model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project='./정재문/runs',      # <-- 1. 최상위 폴더 설정
        name='pill_detection_yolov8s', # 학습 결과가 저장될 폴더 이름
        device='mps'  # <-- ★★★ apple GPU 사용 (정재문 전용) ★★★
    )
    
    print("Training complete.")
    print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()