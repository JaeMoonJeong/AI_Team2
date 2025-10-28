from ultralytics import YOLO
import os

def main():
    # --- 1. 모델 로드 ---
    # 'yolov10s.pt'에서 'yolov9s.pt'로 변경
    # YOLOv9 모델을 처음 사용하는 경우, Ultralytics가 자동으로 가중치를 다운로드합니다.
    print("Loading YOLOv9s model...")
    model = YOLO('yolov9s.pt') 

    # --- 2. 데이터셋 YAML 파일 경로 ---
    # data.yaml 파일의 상대 경로
    data_yaml_path = 'data.yaml'

    # --- 3. 모델 학습 ---
    print("Starting model training with YOLOv9s...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project='./정재문/runs',  # <-- 1. 최상위 폴더 설정 (이전 '정재문' 폴더 제거)
        name='pill_detection_yolov9s', # <-- 2. 'yolov9s'로 이름 변경
        device='mps', # <-- Apple Silicon (M1/M2/M3) GPU 사용
        
        # --- (선택) 성능 향상을 위한 Augmentation 파라미터 ---
        # mosaic=1.0,  # Mosaic augmentation (과적합 방지)
        # mixup=0.1,   # Mixup augmentation (과적합 방지)
        # degrees=15,  # Random rotation
        # hsv_s=0.5,   # Hue-Saturation-Value augmentation
        # hsv_v=0.5
    )
    
    print("Training complete.")
    print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()

