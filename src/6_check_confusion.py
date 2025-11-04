import os
import numpy as np
from ultralytics import YOLO
import torch # (!!!) torch 임포트 추가

# --- [필수] 사용자 설정 ---
# (가장 성능이 좋았던 V4 훈련 결과 폴더)
RUNS_DIR = './runs_local/'
LATEST_RUN_NAME = 'pill_detection_yolov8s_v4_800' # (!!!) 0.98390점 모델 폴더

# (V4 훈련에 사용한 데이터셋)
DATA_YAML_NAME = 'DATASET_YOLO_v4_perfect/data.yaml'

DEVICE_SETTING = 'cuda:0'
TOP_N_ERRORS = 5 # 상위 5개 오류만 표시
# -------------------------

def check_top_confusions():
    """
    (수정됨) 최종 훈련된 모델의 혼동 행렬을 분석하여
    가장 많이 헷갈려 한 클래스 목록을 출력합니다.
    """
    print(f"--- 📈 혼동 행렬(Confusion Matrix) 분석 시작 ---")

    # 1. 모델 가중치 경로 설정
    model_path = os.path.join(RUNS_DIR, LATEST_RUN_NAME, 'weights/best.pt')
    # 2. 데이터 YAML 경로 설정
    data_path = os.path.join('.', DATA_YAML_NAME) # (예: ./DATASET_YOLO_v4_perfect/data.yaml)
    
    if not os.path.exists(model_path):
        print(f"🚨 오류: 모델 파일({model_path})을 찾을 수 없습니다.")
        return
    if not os.path.exists(data_path):
        print(f"🚨 오류: data.yaml 파일({data_path})을 찾을 수 없습니다.")
        return

    # 3. 모델 로드 및 평가 실행
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    print(f"데이터셋({data_path})으로 검증(Validation)을 다시 실행하여 원본 데이터를 가져옵니다...")
    # model.val()을 실행하면 혼동 행렬이 계산됨
    metrics = model.val(data=data_path, device=DEVICE_SETTING, verbose=False) 
    
    # 4. 혼동 행렬 원본 데이터 및 클래스 이름 가져오기
    if not hasattr(metrics, 'confusion_matrix'):
        print("🚨 오류: 'metrics' 객체에서 혼동 행렬을 찾을 수 없습니다.")
        return
        
    matrix = metrics.confusion_matrix.matrix # (74, 74) 크기의 2D Numpy 배열 (배경 포함)
    names = metrics.names                 # {0: '알약A', ..., 72: '알약Z'} (73개)
    
    if matrix is None or names is None:
        print("🚨 오류: 혼동 행렬 데이터가 비어있습니다.")
        return

    # 5. 가장 많이 헷갈린 Top N 찾기
    
    # (1) 대각선(정답)을 0으로 만들어 오류만 남김
    np.fill_diagonal(matrix, 0)
    
    # (2) 가장 큰 값부터 순서대로 정렬된 인덱스를 찾음
    indices = np.unravel_index(np.argsort(matrix, axis=None), matrix.shape)
    
    print(f"\n--- 📈 모델이 가장 많이 헷갈려 한 Top {TOP_N_ERRORS} (배경 제외) ---")
    
    nc = len(names) # 실제 클래스 개수 (예: 73)
    errors_found = 0
    
    # (3) (!!!) [수정됨] 모든 오류를 거꾸로 순회 (TypeError 및 KeyError 동시 해결)
    for idx in reversed(range(len(indices[0]))):
        
        # 찾으려는 Top N 개수만큼 찾았으면 중단
        if errors_found >= TOP_N_ERRORS:
            break

        pred_idx = indices[0][idx]   # 모델이 '예측'한 클래스 ID
        actual_idx = indices[1][idx] # '실제' 정답 클래스 ID
        count = matrix[pred_idx, actual_idx] # 헷갈린 횟수
        
        # 헷갈린 횟수가 0이면 (오류가 아니면) 중단
        if count == 0:
            break 

        # (!!!) [수정됨] 예측 또는 정답이 '배경' 클래스(ID 73)이면 건너뛰기
        if pred_idx >= nc or actual_idx >= nc:
            continue # 배경 클래스와의 혼동은 무시하고 다음 오류를 찾음

        # 이제 pred_idx와 actual_idx는 0~72 사이의 유효한 값이 보장됨
        actual_name = names[actual_idx] 
        pred_name = names[pred_idx]     
        
        print(f"  {int(count)}회: (정답) '{actual_name}'을(를) (예측) '{pred_name}'(으)로 잘못 예측함")
        errors_found += 1 # 유효한 오류 횟수 카운트

if __name__ == "__main__":
    check_top_confusions()