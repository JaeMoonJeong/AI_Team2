# 원본 Train/Val 분할 스크립트
# '깨끗한' 이미지 638개 분량의 Bbox 2402개를
# '.\preprocessed_data\train_annotations_perfect_clean.csv' 파일로 저장했습니다.

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- [필수] 사용자 설정 ---
BASE_PROJECT_PATH = '.' # 현재 프로젝트 폴더 (AI_Team2)
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')

# [입력] (!!!) 방금 생성한 '깨끗한' 마스터 CSV
# MASTER_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'train_annotations_perfect_clean.csv') # (!!!) 파일명 확인!
MASTER_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'train_annotations_v4_PERFECT_CLEAN.csv') # (!!!) 파일명 확인!

# [입력] CSV 컬럼명 설정
FILENAME_COLUMN = 'image_file'  # (!!!) 깨끗한 CSV의 파일명 컬럼 확인!
CLASSNAME_COLUMN = 'class_name' # (!!!) 깨끗한 CSV의 클래스명 컬럼 확인!

# 분할 비율
VAL_SIZE = 0.2
RANDOM_STATE = 42

# [출력] (!!!) 생성될 '깨끗한' Train/Val CSV 파일 이름
OUTPUT_DIR = PREPROCESSED_DIR
TRAIN_CSV_NAME = 'train_set_v4_clean.csv' # (!!!) 새 이름 추천
VAL_CSV_NAME = 'val_set_v4_clean.csv'   # (!!!) 새 이름 추천!
# --------------------

def split_clean_dataset_stratified():
    """
    (수정됨) '깨끗한' CSV를 읽어 클래스 분포를 고려하여
    깨끗한 Train/Validation set CSV를 생성합니다.
    """
    try:
        df_master = pd.read_csv(MASTER_CSV_PATH)
    except FileNotFoundError:
        print(f"🚨 오류: '{MASTER_CSV_PATH}' 파일을 찾을 수 없습니다.")
        return
    print(f"'{MASTER_CSV_PATH}' 파일 읽기 완료.")

    if FILENAME_COLUMN not in df_master.columns or CLASSNAME_COLUMN not in df_master.columns:
        print(f"🚨 오류: CSV에 '{FILENAME_COLUMN}' 또는 '{CLASSNAME_COLUMN}' 컬럼이 없습니다.")
        print(f"   컬럼명 변수를 확인하세요. 현재 컬럼: {list(df_master.columns)}")
        return

    print("클래스 희귀도 계산 중...")
    class_counts = df_master[CLASSNAME_COLUMN].value_counts()
    image_classes = df_master.groupby(FILENAME_COLUMN)[CLASSNAME_COLUMN].apply(list)

    def get_rarest_class(classes):
        return min(classes, key=lambda c: class_counts.get(c, float('inf')))

    image_stratify_key = image_classes.apply(get_rarest_class)
    unique_images = image_stratify_key.index
    stratify_values = image_stratify_key.values

    print(f"총 {len(unique_images)}개의 고유 이미지를 '클래스 희귀도' 기준으로 분할합니다.")

    try:
        train_files, val_files = train_test_split(
            unique_images, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=stratify_values
        )
    except ValueError as e:
        print(f"\n--- 🚨 층화 분할 오류 --- {e}")
        print("   일반 랜덤 분할로 대체합니다...")
        train_files, val_files = train_test_split(unique_images, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    print(f"분할 결과 -> Train: {len(train_files)}개 이미지, Validation: {len(val_files)}개 이미지")

    # 데이터프레임 분리 및 저장
    df_train = df_master[df_master[FILENAME_COLUMN].isin(train_files)]
    df_val = df_master[df_master[FILENAME_COLUMN].isin(val_files)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_output_path = os.path.join(OUTPUT_DIR, TRAIN_CSV_NAME)
    val_output_path = os.path.join(OUTPUT_DIR, VAL_CSV_NAME)

    df_train.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    df_val.to_csv(val_output_path, index=False, encoding='utf-8-sig')

    print(f"\n--- ✅ 깨끗한 Train/Val 분할 완료 ---")
    print(f"깨끗한 훈련셋 CSV: '{train_output_path}' (Bbox {len(df_train)}개)")
    print(f"깨끗한 검증셋 CSV: '{val_output_path}' (Bbox {len(df_val)}개)")

    train_classes = set(df_train[CLASSNAME_COLUMN].unique())
    missing_in_train = set(class_counts.index) - train_classes
    if not missing_in_train:
        print("✅ (성공) 모든 클래스가 훈련셋에 포함되었습니다.")
    else:
        print(f"🚨 (경고) 훈련셋에 누락된 클래스: {missing_in_train}")

# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    split_clean_dataset_stratified()