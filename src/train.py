import os
import torch
from ultralytics import YOLO

# --- 설정 (Configuration) ---

# 1. data.yaml 파일의 *절대 경로*
#    제공해주신 'data.yaml' 파일의 실제 Google Drive 경로를 지정합니다.
DATA_YAML_PATH = "/content/drive/MyDrive/데이터_인프런/AI프로젝트_2팀/DATASET_YOLO_v1/data.yaml"

# 2. 학습 설정
YOLO_MODEL = "yolov8l.pt"  # 사용할 YOLO 모델 (예: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)
EPOCHS = 50               # 총 학습 에포크
IMG_SIZE = 640            # 입력 이미지 크기
BATCH_SIZE = 16           # 배치 사이즈 (GPU 메모리 상태에 따라 조절)

# 3. 결과 저장 설정
# 학습 결과(가중치, 로그 등)가 저장될 경로입니다.
PROJECT_NAME = "Pill_Detection_Project" # 결과가 저장될 상위 폴더 이름
RUN_NAME = f"{YOLO_MODEL.split('.')[0]}_epochs{EPOCHS}_batch{BATCH_SIZE}" # 이번 학습을 구분할 하위 폴더 이름
# ------------------------------

def train_model():
    """
    data.yaml 파일을 참조하여 YOLO 모델 학습을 수행합니다.
    """
    
    # 1. GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 현재 사용 중인 디바이스: {device}")
    
    if not torch.cuda.is_available():
        print("[WARNING] CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다. (시간이 매우 오래 걸릴 수 있습니다)")

    # 2. data.yaml 파일 존재 여부 확인
    if not os.path.exists(DATA_YAML_PATH):
        print(f"[ERROR] data.yaml 파일을 찾을 수 없습니다: {DATA_YAML_PATH}")
        print("스크립트 상단의 'DATA_YAML_PATH' 변수 경로를 다시 확인해주세요.")
        return

    print(f"[INFO] data.yaml 파일 로드 성공: {DATA_YAML_PATH}")
    
    # 3. YOLO 모델 로드
    # YOLO_MODEL에 지정된 사전 학습된 가중치를 로드합니다.
    model = YOLO(YOLO_MODEL)
    model.to(device) # 모델을 지정된 디바이스(GPU 또는 CPU)로 이동

    # 4. 모델 학습 시작
    print(f"[START] 모델 학습을 시작합니다...")
    print(f"  - 모델: {YOLO_MODEL}")
    print(f"  - 에포크: {EPOCHS}")
    print(f"  - 배치 사이즈: {BATCH_SIZE}")
    print(f"  - 결과 저장 경로: {os.path.join(PROJECT_NAME, RUN_NAME)}")

    # model.train() 함수가 DATA_YAML_PATH를 읽어
    # nc, names, train, val 경로를 모두 자동으로 인식합니다.
    results = model.train(
        data=DATA_YAML_PATH,    # *핵심: data.yaml 파일 경로 지정*
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        device=device.type,     # 'cuda' 또는 'cpu'
        exist_ok=True           # 동일한 이름의 RUN_NAME이 있어도 덮어쓰기 허용
    )
    
    print("[COMPLETE] 모델 학습이 완료되었습니다.")
    
    # 학습이 완료되면 'best.pt' 가중치 파일이 저장됩니다.
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
    print(f"[INFO] 최적 가중치 파일이 다음 경로에 저장되었습니다:")
    print(f"{best_model_path}")

if __name__ == "__main__":
    print("=== YOLOv8 모델 학습 프로그램 시작 ===")
    train_model()
    print("=== YOLOv8 모델 학습 프로그램 종료 ===")