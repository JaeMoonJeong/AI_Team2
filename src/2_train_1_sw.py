from ultralytics import YOLO
import os

def main():
    # --- 1. 모델 로드 ---
    # Transformer 기반의 RT-DETR-L (Large) 모델을 로드합니다.
    # 처음 사용하는 경우, 가중치를 자동으로 다운로드합니다.
    print("Loading RT-DETR-L (Transformer) model...")
    model = YOLO('rtdetr-l.pt') 

    # --- 2. 데이터셋 YAML 파일 경로 ---
    # 동일한 data.yaml 파일을 사용합니다.
    data_yaml_path = 'data.yaml'

    # --- 3. 모델 학습 ---
    print("Starting model training with RT-DETR-L (imgsz=640, Heavy Augmentation)...")
    results = model.train(
        data=data_yaml_path,
        epochs=150,                     # 훈련 횟수 (최대)
        imgsz=640,                      # 이미지 해상도 (v9와 동일하게)
        batch=8,                        # <-- (v9s보다 무거우므로 배치 8로 시작. 메모리 부족 시 4로 줄이세요)
        project='./runs',               # 저장 경로
        name='pill_detection_rtdetr-l_Aug_640', # <-- 실험 이름 변경
        device='cpu', # <-- Apple Silicon (M1/M2/M3) GPU 사용
        patience=30,                    # 조기 종료 설정 유지
        
        # --- (v9s와 동일) Augmentation 파라미터 ---
        #mosaic=1.0,  # Mosaic augmentation (과적합 방지)
        #mixup=0.1,   # Mixup augmentation (과적합 방지)
        #degrees=15,  # Random rotation
        #hsv_s=0.7,   # Hue-Saturation (색상) 증강
        #hsv_v=0.4    # Value (밝기) 증강
    )
    
    print("Training complete.")
    print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
