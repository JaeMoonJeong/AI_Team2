# 기본 깨끗한 데이터(638개)에 오프라인 증강을 적용해보는 스크립트
import albumentations as A
import pandas as pd
import cv2
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image # Pillow 사용

# --- [필수] 사용자 설정 ---

# 1. 클래스별 최소 목표 Bbox 개수
TARGET_PER_CLASS = 120 # 목표 개수 (예: 120)

# 2. 기본 경로
BASE_PROJECT_PATH = '.' # 현재 프로젝트 폴더 (AI_Team2)
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')
# 베이스라인 훈련셋 이미지 폴더 (깨끗한 510개 이미지 복사본)
ORIGINAL_IMAGE_DIR = os.path.join(BASE_PROJECT_PATH, 'DATASET_YOLO_v4_baseline', 'images', 'train')

# 3. [입력] 깨끗한 '훈련용' CSV
TRAIN_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'train_set_v4_clean.csv') # (!!!) 파일명 확인!

# 4. CSV 컬럼명
FILENAME_COL = 'image_file'
CLASSNAME_COL = 'class_name'
XMIN_COL = 'x_min'
YMIN_COL = 'y_min'
XMAX_COL = 'x_max'
YMAX_COL = 'y_max'

# 5. [출력] 새로운 파일/폴더 이름 (V? - Perfect)
AUG_IMAGE_DIR = os.path.join(PREPROCESSED_DIR, 'augmented_images_v4_perfect') # (!!!) 새 이름!
FINAL_AUG_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'train_aug_master_v4_perfect.csv') # (!!!) 새 이름!

# -------------------------------------------------

# (최소 증강 버전 함수)
def get_strong_augmentations_v2():
    """
    (수정됨) 매우 약한 증강만 사용하는 최소 버전
    """
    # print("--- [DEBUG] 최소 증강 파이프라인 사용 중 ---") # 필요 시 주석 해제
    return A.Compose([
        A.HorizontalFlip(p=0.5), # 좌우 반전
        A.Rotate(limit=5, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0), # 약한 회전(±5도)
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1 # Bbox 가시성이 10% 미만이면 제거
    ))

# --- 메인 스크립트 ---
if __name__ == "__main__":
    print("--- 🚀 오프라인 증강 스크립트 (V4 Perfect) 시작 ---")
    print(f" (i) 원본 이미지 소스: {ORIGINAL_IMAGE_DIR}")
    print(f" (i) 원본 훈련 CSV: {TRAIN_CSV_PATH}")

    os.makedirs(AUG_IMAGE_DIR, exist_ok=True)

    try:
        df_train_orig = pd.read_csv(TRAIN_CSV_PATH) # 깨끗한 훈련셋(510개 분량) 로드
    except FileNotFoundError:
        print(f"🚨 오류: '{TRAIN_CSV_PATH}'를 찾을 수 없습니다.")
        exit()
    print(f"원본 훈련 데이터 로드 완료. (Bbox {len(df_train_orig)}개)")

    transform = get_strong_augmentations_v2()
    all_class_names = list(df_train_orig[CLASSNAME_COL].unique())
    class_to_id_map = {name: i for i, name in enumerate(all_class_names)}
    id_to_class_map = {i: name for name, i in class_to_id_map.items()}

    class_counts = df_train_orig[CLASSNAME_COL].value_counts()
    rare_classes = class_counts[class_counts < TARGET_PER_CLASS]

    if len(rare_classes) == 0:
        print(f"✅ 모든 클래스가 목표치({TARGET_PER_CLASS}개) 이상입니다.")
        df_train_orig.to_csv(FINAL_AUG_CSV_PATH, index=False, encoding='utf-8-sig')
        exit()

    print(f"총 {len(rare_classes)}개의 희귀 클래스를 식별했습니다 (목표: {TARGET_PER_CLASS}개).")
    new_rows = []

    for class_name in tqdm(rare_classes.index, desc="희귀 클래스 증강 중"):
        needed = TARGET_PER_CLASS - rare_classes[class_name]
        images_with_class = df_train_orig[df_train_orig[CLASSNAME_COL] == class_name][FILENAME_COL].unique()

        if len(images_with_class) == 0:
            tqdm.write(f"Warning: {class_name} 원본 이미지를 찾을 수 없음.")
            continue

        success_count = 0
        for i in range(needed):
            if i > 50 and success_count == 0:
                tqdm.write(f"--- [DEBUG] {class_name}: 50번 실패. 중단.")
                break

            img_filename = random.choice(images_with_class)
            # (!!!) 수정된 경로에서 이미지 찾기
            img_path = os.path.join(ORIGINAL_IMAGE_DIR, img_filename) 

            try:
                pil_image = Image.open(img_path).convert('RGB')
                image = np.array(pil_image)
            except Exception as e:
                tqdm.write(f"--- [DEBUG] {class_name} 이미지 읽기 실패: {img_path}. {e}")
                continue

            # --- 4.2. Bbox 로드 및 유효성 검사 (강화됨) ---
            records = df_train_orig[df_train_orig[FILENAME_COL] == img_filename]
            
            valid_bboxes = []
            valid_labels = []
            
            for _, row in records.iterrows():
                xmin, ymin, xmax, ymax = row[XMIN_COL], row[YMIN_COL], row[XMAX_COL], row[YMAX_COL]
                
                # [검사 1] NaN (Not a Number) 값이 있는지 확인
                if np.isnan([xmin, ymin, xmax, ymax]).any():
                    tqdm.write(f"--- [DEBUG] {class_name} (img: {img_filename}): NaN Bbox 발견. 건너뜁니다.")
                    continue
                # [검사 2] xmin < xmax, ymin < ymax 인지 확인
                if xmin >= xmax or ymin >= ymax:
                    tqdm.write(f"--- [DEBUG] {class_name} (img: {img_filename}): 잘못된 Bbox 좌표(min>=max) 발견. 건너뜁니다.")
                    continue
                    
                # (!!!) [추가된 검사 3] 좌표값이 1.0 이하인 비정상적인 값인지 확인
                # (6.72 에러의 원인인 0.48 같은 값을 여기서 걸러냄)
                if xmin <= 1.0 or ymin <= 1.0 or xmax <= 1.0 or ymax <= 1.0:
                    tqdm.write(f"--- [DEBUG] {class_name} (img: {img_filename}): 정규화된 값으로 의심되는 Bbox(값 <= 1.0) 발견. 건너뜁니다.")
                    continue
                # --- 검사 끝 ---
                    
                # 모든 검사를 통과한 '유효한' 픽셀 좌표 Bbox만 추가
                valid_bboxes.append([xmin, ymin, xmax, ymax])
                valid_labels.append(class_to_id_map[row[CLASSNAME_COL]])

            if len(valid_bboxes) == 0:
                continue

            try:
                augmented = transform(image=image, bboxes=valid_bboxes, labels=valid_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels_ids = augmented['labels']
            except Exception as e:
                tqdm.write(f"--- [DEBUG] {class_name} 증강 실패: {img_filename}. {e}")
                continue

            if isinstance(aug_image, np.ndarray) and len(aug_bboxes) > 0:
                try:
                    # (!!!) [수정됨] 한글 파일명 생성 (이전과 동일)
                    safe_class_name = class_name.replace(' ', '_').replace('/', '_')
                    new_filename = f"{safe_class_name}_aug_{i}_{img_filename}"
                    new_img_path = os.path.join(AUG_IMAGE_DIR, new_filename)
                    
                    # --- (!!!) [수정됨] 한글 경로 호환 저장 방식 ---
                    # 1. 이미지를 RGB -> BGR로 변환
                    image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    # 2. 이미지를 메모리 버퍼(.png 형식)로 인코딩
                    is_success, im_buf_arr = cv2.imencode(".png", image_bgr)
                    
                    if not is_success:
                        raise ValueError("cv2.imencode 실패")
                        
                    # 3. 파이썬의 open() 함수로 파일에 쓰기 (UTF-8 경로 지원)
                    with open(new_img_path, "wb") as f_buf:
                        f_buf.write(im_buf_arr.tobytes())
                    # --- 수정 끝 ---
                    success_count += 1

                    for bbox, label_id in zip(aug_bboxes, aug_labels_ids):
                        xmin, ymin, xmax, ymax = map(int, bbox)
                        aug_class_name = id_to_class_map[label_id]
                        new_rows.append({
                            FILENAME_COL: new_filename, CLASSNAME_COL: aug_class_name,
                            XMIN_COL: xmin, YMIN_COL: ymin, XMAX_COL: xmax, YMAX_COL: ymax
                        })
                except Exception as e:
                    tqdm.write(f"--- [DEBUG] {class_name} 파일 쓰기 실패: {new_filename}. {e}")

    print("증강 루프 완료. 최종 CSV 파일 생성 중...")
    df_augmented = pd.DataFrame(new_rows)
    df_final_master = pd.concat([df_train_orig, df_augmented], ignore_index=True)
    df_final_master.to_csv(FINAL_AUG_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n--- ✅ 오프라인 증강 (V4 Perfect) 완료! ---")
    print(f"새로운 훈련 마스터 CSV: {FINAL_AUG_CSV_PATH} (총 Bbox: {len(df_final_master)})")
    if os.path.exists(AUG_IMAGE_DIR):
        print(f"새로운 증강 이미지 폴더: {AUG_IMAGE_DIR} (이미지 {len(os.listdir(AUG_IMAGE_DIR))}개 생성됨)")
    else:
         print(f"새로운 증강 이미지 폴더 ({AUG_IMAGE_DIR})가 생성되지 않았습니다.")
         
    print("\n--- 증강 후 클래스별 Bbox 개수 (Bottom 5) ---")
    print(df_final_master[CLASSNAME_COL].value_counts().tail(5))