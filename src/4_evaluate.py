from ultralytics import YOLO
import glob  # 파일/폴더 검색을 위한 라이브러리
import os    # 파일/폴더 정보를 다루기 위한 라이브러리

# --- [필수] 사용자 설정 ---

# 1. (!!! 수정 !!!) 훈련 결과가 저장된 기본 폴더 경로
#    (2_train.py의 'project' 설정과 동일해야 함)
TRAIN_RESULTS_BASE_DIR = "./runs_local/" # <-- 훈련 결과 폴더 경로

# 2. (!!! 수정 !!!) 훈련 시 사용했던 data.yaml 파일 경로
#    (스크립트 위치 기준 상대 경로 또는 절대 경로)
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
DATA_YAML_PATH = os.path.join(project_root, 'DATASET_YOLO_v4_perfect', 'data.yaml') # <-- 기준선 데이터셋 yaml

# 3. GPU 설정 (!!! 수정 !!!)
DEVICE_SETTING = 'cuda:0' # 또는 0 (NVIDIA GPU 사용)


def get_latest_run_directory(base_dir="./runs_local/"):
    """
    base_dir(기본 검색 경로) 안에서 가장 최근에 수정된 폴더(run)의
    경로를 찾아서 반환합니다.
    """
    # base_dir 안의 모든 하위 폴더 목록을 가져옵니다.
    list_of_dirs = glob.glob(os.path.join(base_dir, '*/'))

    if not list_of_dirs:
        print(f"오류: '{base_dir}' 경로에서 학습 폴더를 찾을 수 없습니다.")
        return None

    # 폴더들을 '마지막 수정 시간(mtime)'을 기준으로 정렬합니다.
    # max() 함수를 사용해 가장 최근(시간 값이 가장 큰) 폴더를 찾습니다.
    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    print(f"가장 최신 학습 폴더를 찾았습니다: {latest_dir}")
    return latest_dir

def main():
    # 1. ★★★ (자동화) 가장 최신 학습 폴더 찾기 ★★★
    latest_run_dir = get_latest_run_directory(TRAIN_RESULTS_BASE_DIR)

    if latest_run_dir is None:
        print("평가를 중단합니다.")
        return

    # 2. ★★★ (자동화) 'best.pt' 경로 조합 ★★★
    trained_model_path = os.path.join(latest_run_dir, 'weights/best.pt')

    # 2. ★★★ data.yaml 경로 확인 ★★★
    # 2_train.py에서 사용한 것과 동일한 경로
    # data_yaml_path = './data.yaml'

    # 1. 상위 프로젝트 폴더: project_dir = "./정재문/runs/"
    # project_dir = os.path.dirname(latest_run_dir.rstrip('/'))
    # 2. 현재 학습 이름: run_name = "pill_detection_yolov8s"
    # run_name = os.path.basename(latest_run_dir.rstrip('/'))

    # 3. 학습된 모델 로드
    print(f"Loading trained model from: {trained_model_path}")
    model = YOLO(trained_model_path)

    # 4. 모델 성능 평가 (Validation)
    # data.yaml에 정의된 'val' 세트를 사용하여 mAP를 계산합니다.
    print("Evaluating model performance on validation set...")
    metrics = model.val(
        data=DATA_YAML_PATH,
        device=DEVICE_SETTING,  # GPU 사용
        split='val',    # Validation set 사용 명시
        project=latest_run_dir, # 평가 결과도 해당 run 폴더 안에 저장
        name='evaluation_v4_perfect_800' # 평가 결과 저장 폴더 이름 
    )

    # 5. ★★★ mAP 값 추출 및 출력 ★★★
    print("\n--- Model Performance Metrics ---")

    # mAP50 (IoU 0.5일 때의 mAP)
    # metrics.box.map50는 0.0 ~ 1.0 사이의 값이므로 100을 곱해 %로 만듭니다.
    map50 = metrics.box.map50
    print(f"mAP@50 (IoU=0.5): {map50 * 100:.2f}%")

    # mAP50-95 (IoU 0.5~0.95 평균 mAP) - 이게 COCO 표준
    map50_95 = metrics.box.map
    print(f"mAP@50-95 (Standard COCO): {map50_95 * 100:.2f}%")

    print("---------------------------------")

    # "mAP 93"이라는 값이 mAP50을 의미했다면,
    # 터미널에 "mAP@50 (IoU=0.5): 93.00%" (예시) 처럼 출력될 것입니다.

if __name__ == '__main__':
    main()