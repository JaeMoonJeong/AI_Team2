import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image # 이미지 유효성 검사까지!

# --- [필수] 사용자 설정 ---
BASE_PROJECT_PATH = '.' # 현재 프로젝트 폴더 (AI_Team2)
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')
ORIGINAL_DIR = os.path.join(BASE_PROJECT_PATH, 'ORIGINAL')

# [입력 1] (!!!) 실제 이미지 파일이 있는 폴더
SOURCE_IMAGE_DIR = os.path.join(ORIGINAL_DIR, 'train_images') # <-- ★★★ 실제 이미지 위치 ★★★

# [입력 2] (!!!) 1489개 '전체' JSON 파일이 들어있는 폴더
SOURCE_ANNOTATIONS_DIR = os.path.join(ORIGINAL_DIR, 'train_annotations') 

# [입력 3] '누락된' 850개 이미지 목록 CSV
MISSING_REPORT_PATH = os.path.join(PREPROCESSED_DIR, 'missing_annotations_report.csv')

# [출력] (!!!) 진짜 '깨끗한' 데이터만 저장할 새 CSV
CLEAN_ANNOTATIONS_PATH = os.path.join(PREPROCESSED_DIR, 'train_annotations_v4_PERFECT_CLEAN.csv') # <-- ★★★ 새 이름!

REPORT_FILENAME_COL = 'image'
IMAGE_WIDTH = 976
IMAGE_HEIGHT = 1280
# -------------------------

def create_clean_master_csv_v4():
    """
    (수정됨) 1. 누락 리포트 제외 + 2. Bbox 이상치 제거 + 3. ★실제 파일 존재/유효성 확인★
    """
    print("--- 🧼 '깨끗한' 마스터 CSV 생성 (v4 - 파일 존재/유효성 확인) 시작 ---")
    
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
    outlier_bbox_count = 0
    file_invalid_count = 0 # (!!!) 파일 없음 또는 유효하지 않음 카운트

    for file_path in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']: continue
            image_info = data['images'][0]
            image_filename = image_info.get('file_name')
            if not image_filename: continue

            # [필터 1] 누락 리포트에 있는지 확인
            if image_filename in bad_image_list:
                continue 

            # (!!!) [필터 2 - 추가됨] 실제 이미지 파일이 'train_images' 폴더에 존재하는지 확인
            image_full_path = os.path.join(SOURCE_IMAGE_DIR, image_filename)
            try:
                # Pillow로 이미지를 열어 유효성(손상 여부)까지 확인
                img = Image.open(image_full_path)
                img.verify() # 파일 헤더 검사
                img.close() # 파일 다시 열기 위해 닫기
                
                # (추가) 이미지 파일 크기가 0바이트 이상인지 확인
                if os.path.getsize(image_full_path) == 0:
                     raise ValueError("파일 크기가 0바이트입니다.")
                     
            except Exception as e:
                # tqdm.write(f"--- [DEBUG] 유효하지 않은 이미지 파일: {image_filename}. {e}") # 필요 시 주석 해제
                file_invalid_count += 1
                continue # 파일이 없거나, 손상되었거나, 0바이트이므로 건너뜀

            # --- 이 아래는 파일이 존재하고 유효한 '깨끗한' 이미지 ---
            processed_clean_images.add(image_filename)
            categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
            
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox')
                cat_id = ann.get('category_id')
                if not (bbox and len(bbox) == 4 and cat_id in categories): continue

                class_name = categories[cat_id]
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h

                # [필터 3] Bbox 이상치 검사
                if not (0 <= xmin < IMAGE_WIDTH and 0 <= ymin < IMAGE_HEIGHT and
                        0 < xmax <= IMAGE_WIDTH and 0 < ymax <= IMAGE_HEIGHT and w > 0 and h > 0):
                    outlier_bbox_count += 1
                    continue
                
                clean_bbox_rows.append({
                    'image_file': image_filename, 'class_name': class_name,
                    'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax
                })
        except Exception as e:
            tqdm.write(f"Warning: {os.path.basename(file_path)} 처리 오류: {e}")

    if not clean_bbox_rows:
        print("🚨 오류: '깨끗한' Bbox 정보를 하나도 추출하지 못했습니다.")
        return

    df_clean_annotations = pd.DataFrame(clean_bbox_rows)
    df_clean_annotations.to_csv(CLEAN_ANNOTATIONS_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n--- ✅ 필터링 완료! ---")
    print(f"(!!!) 존재하지 않거나 유효하지 않은 이미지 파일 {file_invalid_count}개를 건너뛰었습니다.")
    print(f"(!!!) Bbox 좌표 이상치 총 {outlier_bbox_count}개를 발견하고 제거했습니다.")
    print(f"최종 '깨끗한' 이미지 {len(processed_clean_images)}개 분량의 Bbox {len(df_clean_annotations)}개를")
    print(f"'{CLEAN_ANNOTATIONS_PATH}' 파일로 덮어썼습니다.")
    print("\n다음 단계로, 이 새 CSV 파일을 사용해 '원본 Train/Val 분할' 스크립트를 실행하세요.")

if __name__ == "__main__":
    create_clean_master_csv_v4()