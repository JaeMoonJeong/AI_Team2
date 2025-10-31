import os
import pandas as pd
import cv2
import yaml
import shutil
from tqdm import tqdm
from PIL import Image # Pillow 사용
import numpy as np # Numpy 사용

# --- [필수] 사용자 설정 (V2 Clean) ---
BASE_PROJECT_PATH = '.'
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')

# [입력 1] 원본 이미지 폴더
ORIGINAL_IMAGE_DIR = os.path.join(BASE_PROJECT_PATH, 'ORIGINAL', 'train_images')
# [입력 2] V2 증강 이미지 폴더 (clean 버전)
# AUG_IMAGE_DIR = os.path.join(PREPROCESSED_DIR, 'augmented_images_v2_clean') # (!!!) 이름 확인!

# [입력 3] V2 증강된 훈련 마스터 CSV (clean 버전)
TRAIN_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'train_set_v4_clean.csv') # (!!!) 이름 확인!
# [입력 4] V1 원본에서 분리된 깨끗한 검증 CSV (!!!)
VAL_CSV_PATH = os.path.join(PREPROCESSED_DIR, 'val_set_v4_clean.csv') # (!!!) 이름 확인!

# [출력] V2 최종 YOLO 데이터셋 루트 폴더
YOLO_DATASET_ROOT = os.path.join(BASE_PROJECT_PATH, 'DATASET_YOLO_v4_baseline') # (!!!) 새 폴더 이름!

# CSV 컬럼명 설정
FILENAME_COL = 'image_file'
CLASSNAME_COL = 'class_name'
XMIN_COL = 'x_min'
YMIN_COL = 'y_min'
XMAX_COL = 'x_max'
YMAX_COL = 'y_max'
# --------------------

# (함수 동일) YOLO Bbox 변환
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x, w, y, h = x*dw, w*dw, y*dh, h*dh
    return (x, y, w, h)

# (수정된 함수) 이미지 로드 및 변환/복사
def process_split(df, class_to_id_map,
                  img_src_dir_orig, img_src_dir_aug,
                  img_dst_dir, label_dst_dir):

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(label_dst_dir, exist_ok=True)
    unique_images = df[FILENAME_COL].unique()

    for filename in tqdm(unique_images):
        # 파일 경로 결정
        is_augmented = img_src_dir_aug is not None and '_aug_' in filename
        img_path_src = os.path.join(img_src_dir_aug if is_augmented else img_src_dir_orig, filename)

        # 이미지 로드 (Pillow + RGB 변환)
        try:
            pil_image = Image.open(img_path_src).convert('RGB')
            image = np.array(pil_image)
            if image is None: raise ValueError("이미지 None")
            h, w, _ = image.shape
        except FileNotFoundError:
             tqdm.write(f"--- [DEBUG] 이미지 못 찾음 (소스 경로 확인): {img_path_src}")
             continue
        except Exception as e:
            tqdm.write(f"Warning: {filename} 이미지 로드 실패 ({img_path_src}). 건너뜁니다. {e}")
            continue

        # 이미지 복사
        img_path_dst = os.path.join(img_dst_dir, filename)
        try:
            # Pillow로 읽었으므로 저장도 Pillow로 (한글 경로 안전)
            pil_image.save(img_path_dst)
        except Exception as e:
             tqdm.write(f"Warning: {filename} 이미지 복사 실패 ({img_path_dst}). 건너뜁니다. {e}")
             continue

        # 라벨 생성
        bboxes = df[df[FILENAME_COL] == filename]
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path_dst = os.path.join(label_dst_dir, label_filename)

        with open(label_path_dst, 'w') as f:
            for _, row in bboxes.iterrows():
                class_name = row[CLASSNAME_COL]
                if class_name not in class_to_id_map:
                    tqdm.write(f"Warning: 알 수 없는 클래스 '{class_name}'. 건너뜁니다.")
                    continue
                class_id = class_to_id_map[class_name]
                box = (row[XMIN_COL], row[YMIN_COL], row[XMAX_COL], row[YMAX_COL])
                yolo_box = convert_to_yolo((w, h), box)
                f.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")

# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    print("--- 🚀 2단계 (V2 Clean) YOLO 데이터셋 생성 시작 ---")

    try:
        df_train = pd.read_csv(TRAIN_CSV_PATH)
        df_val = pd.read_csv(VAL_CSV_PATH)
    except FileNotFoundError as e:
        print(f"🚨 오류: CSV 파일을 찾을 수 없습니다. 경로를 확인하세요. {e}")
        exit()

    # 클래스 ID 맵 생성 (Train + Val 모든 클래스 포함)
    all_class_names_train = set(df_train[CLASSNAME_COL].unique())
    all_class_names_val = set(df_val[CLASSNAME_COL].unique())
    unique_class_names = sorted(list(all_class_names_train.union(all_class_names_val)))
    CLASS_NAMES = list(unique_class_names)
    NC = len(CLASS_NAMES)
    CLASS_TO_ID_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
    print(f"총 {NC}개의 클래스를 찾았습니다 (Train+Val).")

    # 경로 정의
    train_img_dst = os.path.join(YOLO_DATASET_ROOT, 'images', 'train')
    val_img_dst = os.path.join(YOLO_DATASET_ROOT, 'images', 'val')
    train_label_dst = os.path.join(YOLO_DATASET_ROOT, 'labels', 'train')
    val_label_dst = os.path.join(YOLO_DATASET_ROOT, 'labels', 'val')

    # Train 데이터 변환 (원본 + V2 증강 폴더 모두 사용)
    print("Train (V2 Clean) 데이터셋 변환 중...")
    process_split(df_train, CLASS_TO_ID_MAP, ORIGINAL_IMAGE_DIR, None, train_img_dst, train_label_dst)

    # Validation 데이터 변환 (원본 폴더만 사용, 증강 폴더는 None)
    print("Validation (Original) 데이터셋 변환 중...")
    process_split(df_val, CLASS_TO_ID_MAP, ORIGINAL_IMAGE_DIR, None, val_img_dst, val_label_dst)

    # data.yaml 파일 생성 (상대 경로 사용)
    yaml_path = os.path.join(YOLO_DATASET_ROOT, 'data.yaml')
    data_yaml = {'train': './images/train', 'val': './images/val', 'nc': NC, 'names': CLASS_NAMES}
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"\n'data.yaml' 파일 생성 완료: {yaml_path}")
    except Exception as e:
        print(f"🚨 오류: data.yaml 파일 쓰기 실패. {e}")

    print(f"--- ✅ 2단계 (V2 Clean) YOLO 데이터셋 생성 완료! ---")
    print(f"최종 폴더: {YOLO_DATASET_ROOT}")