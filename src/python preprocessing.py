import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np  # (필요시 데이터 분석용으로 남겨둘 수 있음)
import pandas as pd # (필요시 데이터 분석용으로 남겨둘 수 있음)

# --- 설정값 ---
# 원본 데이터 경로
path = os.getcwd()
data_path = os.path.join(path, 'data')
annotations_path = os.path.join(data_path, 'train_annotations')
train_image_path = os.path.join(data_path, "train_images")

# 결과물(YOLO 데이터셋) 경로
OUTPUT_DIR = os.path.join(data_path, 'dataset_yolo')
SPLIT_RATIO = 0.8
# ----------------

def scan_json_files():
    """주석 경로에서 모든 JSON 파일 스캔"""
    json_path_list = []
    first_level = os.listdir(annotations_path)
    for second_level in first_level:
        folder_path = os.path.join(annotations_path, second_level)
        for third_level in os.listdir(folder_path):
            medicine_path = os.path.join(folder_path, third_level)
            for json_file in os.listdir(medicine_path):
                json_path = os.path.join(medicine_path, json_file)
                json_path_list.append(json_path)
    return json_path_list

def convert_bbox(bbox, img_w, img_h):
    """COCO bbox(x, y, w, h) -> YOLO bbox(cx, cy, w, h) [normalized]"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return cx, cy, w_norm, h_norm

def read_json(p):
    """JSON 파일 읽기"""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def process_dataset():
    """메인 전처리 함수"""
    print("[시작] 데이터 변환 시작!")
    json_path = scan_json_files()
    print(f"[확인] JSON 파일 {len(json_path)}개 발견")

    merged_annotations = []
    merged_images = []
    cat_seen = set()

    for jp in tqdm(json_path, desc="JSON 파일 병합 중"):
        data = read_json(jp)
        for annotation in data["annotations"]:
            merged_annotations.append(annotation)
            cat_seen.add(annotation["category_id"])
        for image in data["images"]:
            # 원본 이미지 경로를 추적하기 위해 json_path의 디렉토리 저장 (사용은 안 함)
            # image["json_path"] = os.path.dirname(jp) 
            merged_images.append(image)
    
    # 카테고리 매핑 생성
    category_list = sorted(list(cat_seen))
    category_mapping = {cat_id : idx for idx, cat_id in enumerate(category_list)}
    
    # 이미지 ID를 기준으로 주석 그룹화 (빠른 탐색용)
    annotation_by_image_id = {}
    for annotation in merged_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotation_by_image_id:
            annotation_by_image_id[image_id] = []
        annotation_by_image_id[image_id].append(annotation)

    # YOLO 데이터셋 폴더 생성
    img_train_dir = os.path.join(OUTPUT_DIR, "images/train")
    lbl_train_dir = os.path.join(OUTPUT_DIR, "labels/train")
    img_val_dir = os.path.join(OUTPUT_DIR, "images/val")
    lbl_val_dir = os.path.join(OUTPUT_DIR, "labels/val")
    
    for p in [img_train_dir, lbl_train_dir, img_val_dir, lbl_val_dir]:
        os.makedirs(p, exist_ok=True)

    # Train / Validation 분할
    random.seed(42)
    random.shuffle(merged_images)
    split_len = int(len(merged_images) * SPLIT_RATIO)
    train_images = merged_images[:split_len]
    val_images = merged_images[split_len:]
    print(f"[INFO] Train: {len(train_images)} 장, Val: {len(val_images)} 장")

    # Train 데이터셋 처리
    print("Train 데이터셋 변환 중...")
    for img in tqdm(train_images):
        img_name = img["file_name"]
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        
        # 1. 이미지 복사
        src_img_path = os.path.join(train_image_path, img_name)
        dst_img_path = os.path.join(img_train_dir, img_name)
        shutil.copy(src_img_path, dst_img_path)

        # 2. 라벨 파일 생성
        if img_id in annotation_by_image_id:
            txt_name = img_name.replace(os.path.splitext(img_name)[1], ".txt")
            with open(os.path.join(lbl_train_dir, txt_name), "w") as f:
                for ann in annotation_by_image_id[img_id]:
                    cls_id = category_mapping[ann["category_id"]]
                    bbox_yolo = convert_bbox(ann["bbox"], img_w, img_h)
                    f.write(f"{cls_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

    # Validation 데이터셋 처리
    print("Validation 데이터셋 변환 중...")
    for img in tqdm(val_images):
        img_name = img["file_name"]
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]

        # 1. 이미지 복사
        src_img_path = os.path.join(train_image_path, img_name)
        dst_img_path = os.path.join(img_val_dir, img_name)
        shutil.copy(src_img_path, dst_img_path)

        # 2. 라벨 파일 생성
        if img_id in annotation_by_image_id:
            txt_name = img_name.replace(os.path.splitext(img_name)[1], ".txt")
            with open(os.path.join(lbl_val_dir, txt_name), "w") as f:
                for ann in annotation_by_image_id[img_id]:
                    cls_id = category_mapping[ann["category_id"]]
                    bbox_yolo = convert_bbox(ann["bbox"], img_w, img_h)
                    f.write(f"{cls_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

    # data.yaml 파일 생성 (가장 중요!)
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        # path는 data.yaml 파일 기준 상대 경로 또는 절대 경로
        # 여기서는 절대 경로를 사용
        f.write(f"path: {OUTPUT_DIR}\n") 
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        for k, v in category_mapping.items():
            f.write(f"  {v}: class_{k}\n") # 클래스 이름을 "class_ID"로 저장

    print(f"[완료] 변환 완료 → {OUTPUT_DIR}")
    print(f"[생성] data.yaml → {yaml_path}")
    
    # category_reverse_mapping은 train.py에서 필요 없음
    # yaml 파일이 그 역할을 대신함


if __name__ == "__main__":
    print("=== 데이터 전처리 프로그램 시작 ===")
    process_dataset()
    print("=== 데이터 전처리 프로그램 종료 ===")