import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# --- [필수] 사용자 설정 ---
BASE_PROJECT_PATH = '.' # 현재 프로젝트 폴더 (AI_Team2)
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')
ORIGINAL_DIR = os.path.join(BASE_PROJECT_PATH, 'ORIGINAL')
SOURCE_ANNOTATIONS_DIR = os.path.join(ORIGINAL_DIR, 'train_annotations') 
MISSING_REPORT_PATH = os.path.join(PREPROCESSED_DIR, 'missing_annotations_report.csv')
CLEAN_ANNOTATIONS_PATH = os.path.join(PREPROCESSED_DIR, 'train_annotations_perfect_clean.csv') # (!!!) 이 파일을 덮어쓸 거예요!
REPORT_FILENAME_COL = 'image'
# -------------------------

# (!!!) 이미지 크기 상수 정의
IMAGE_WIDTH = 976
IMAGE_HEIGHT = 1280

def create_clean_master_csv():
    """
    (수정됨) '누락 리포트' 제외 + 'Bbox 이상치' 제거 로직 추가
    """
    print("--- 🧼 '깨끗한' 마스터 CSV 생성 (이상치 제거 v3) 시작 ---")
    
    try:
        df_report = pd.read_csv(MISSING_REPORT_PATH)
        bad_image_list = set(df_report[REPORT_FILENAME_COL].unique())
        print(f"'누락된' 이미지 목록 {len(bad_image_list)}개를 리포트에서 로드했습니다.")
    except Exception as e:
        print(f"🚨 오류: 'missing_annotations_report.csv' 파일 읽기 실패. {e}")
        return

    json_files = []
    print(f"'{SOURCE_ANNOTATIONS_DIR}' 폴더에서 JSON 파일을 검색 중...")
    for root, _, files in os.walk(SOURCE_ANNOTATIONS_DIR):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"🚨 오류: '{SOURCE_ANNOTATIONS_DIR}'에서 JSON 파일을 찾을 수 없습니다.")
        return
    print(f"총 {len(json_files)}개의 JSON 파일 발견. '깨끗한' 데이터 추출을 시작합니다...")

    clean_bbox_rows = []
    processed_clean_images = set()
    outlier_bbox_count = 0 # (!!!) 이상치 카운트

    for file_path in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']: continue
                
            image_info = data['images'][0]
            image_filename = image_info.get('file_name')
            if not image_filename: continue

            if image_filename in bad_image_list:
                continue 

            processed_clean_images.add(image_filename)
            categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
            
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox')
                cat_id = ann.get('category_id')
                
                if not (bbox and len(bbox) == 4 and cat_id in categories):
                    continue

                class_name = categories[cat_id]
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h

                # --- (!!!) [추가됨] Bbox 이상치 검사 (!!!) ---
                if not (0 <= xmin < IMAGE_WIDTH and 
                        0 <= ymin < IMAGE_HEIGHT and
                        0 < xmax <= IMAGE_WIDTH and  # xmax는 너비보다 클 수 없음
                        0 < ymax <= IMAGE_HEIGHT and # ymax는 높이보다 클 수 없음
                        w > 0 and h > 0):
                    
                    # tqdm.write(f"--- [DEBUG] {image_filename}: Bbox 이상치 발견 (제거됨) -> {bbox}") # 필요 시 주석 해제
                    outlier_bbox_count += 1
                    continue # 이 Bbox는 저장하지 않고 건너뜀
                # --- 검사 끝 ---

                clean_bbox_rows.append({
                    'image_file': image_filename,
                    'class_name': class_name,
                    'x_min': xmin,
                    'y_min': ymin,
                    'x_max': xmax,
                    'y_max': ymax
                })

        except Exception as e:
            tqdm.write(f"Warning: {os.path.basename(file_path)} 처리 중 오류 발생: {e}")

    if not clean_bbox_rows:
        print("🚨 오류: '깨끗한' Bbox 정보를 하나도 추출하지 못했습니다.")
        return

    df_clean_annotations = pd.DataFrame(clean_bbox_rows)
    df_clean_annotations.to_csv(CLEAN_ANNOTATIONS_PATH, index=False) # 덮어쓰기
    
    print(f"\n--- ✅ 필터링 완료! ---")
    print(f"(!!!) Bbox 이상치 총 {outlier_bbox_count}개를 발견하고 제거했습니다.")
    print(f"'깨끗한' 이미지 {len(processed_clean_images)}개 분량의 Bbox {len(df_clean_annotations)}개를")
    print(f"'{CLEAN_ANNOTATIONS_PATH}' 파일로 덮어썼습니다.")
    
    print("\n다음 단계로, 이 새 CSV 파일을 사용해 '원본 Train/Val 분할' 스크립트를 [다시] 실행하세요.")

if __name__ == "__main__":
    create_clean_master_csv()