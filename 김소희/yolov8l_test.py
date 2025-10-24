# import 부분

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch        # GPU 확인용
import os
from pathlib import Path
from tqdm import tqdm
import json
import random
import shutil
from ultralytics import YOLO    # 나중에 학습할 때 필요


# GPU 사용 설정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# JSON 파일 스캔
def scan_json_files():
    json_path_list = []
    first_level = os.listdir(annotations_path)  # os.listdir()는 해당 경로의 폴더 및 파일 목록을 리스트로 반환함
    for second_level in first_level:    
        # 첫 번째 폴더 안의 목록 순회
        folder_path = os.path.join(annotations_path, second_level)
        for third_level in os.listdir(folder_path):
            # 두 번째 폴더 안의 목록 순회
            medicine_path = os.path.join(folder_path, third_level)
            for json in os.listdir(medicine_path):
                # JSON 파일들의 전체 경로를 리스트에 추가
                json_path = os.path.join(medicine_path, json)
                json_path_list.append(json_path)
    return json_path_list

path = os.getcwd()
data_path = os.path.join(path, 'data')  # os.path.join()은 경로 합쳐주는 함수

annotations_path = os.path.join(data_path, 'train_annotations') 
train_image_path = os.path.join(data_path, "train_images")
test_image_path = os.path.join(data_path, "test_images")



# 스플릿 비율
SPLIT_RATIO = 0.8
OUTPUT_DIR = os.path.join(data_path, 'dataset_yolo')

def convert_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2

    return cx / img_w, cy / img_h, w / img_w, h / img_h


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("[시작] 데이터 변환 시작!")
    json_path = scan_json_files()
    print(f"[확인] JSON 파일 {len(json_path)}개 발견")

    merged_annotations = []
    merged_images = []
    category_mapping = {}
    cat_seen = set()

    for jp in json_path:
        data = read_json(jp)
        for annotation in data["annotations"]:
            merged_annotations.append(annotation)
            cat_seen.add(annotation["category_id"])
        for image in data["images"]:
            image["json_path"] = os.path.dirname(jp)
            merged_images.append(image)
    
    category_list = sorted(list(cat_seen))
    category_mapping = {cat_id : idx for idx, cat_id in enumerate(category_list)}
    category_reverse_mapping = {v: k for k, v in category_mapping.items()}

    annotation_by_image_id = {}
    for annotation in merged_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotation_by_image_id:
            annotation_by_image_id[image_id] = []
        annotation_by_image_id[image_id].append(annotation)


    # 폴더 생성
    for p in ["images/train", "images/val","labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, p), exist_ok=True)

    # train/val split
    random.seed(42)
    random.shuffle(merged_images)
    split_len = int(len(merged_images) * SPLIT_RATIO)
    train_images = merged_images[:split_len]
    val_images = merged_images[split_len:]
    print(f"[INFO] Train {len(train_images)}, Val {len(val_images)}")


    # train 변환
    for img in tqdm(train_images):
        img_name = img["file_name"]
        img_path = os.path.join(train_image_path, img_name)

        # 이미지 복사
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images/train", img_name))

        # 라벨 저장
        if img["id"] in annotation_by_image_id:
            w, h = img["width"], img["height"]
            with open(os.path.join(OUTPUT_DIR, "labels/train", img_name.replace(os.path.splitext(img_name)[1], ".txt")), "w") as f:
                for ann in annotation_by_image_id[img["id"]]:
                    cls_id = category_mapping[ann["category_id"]]
                    bbox = convert_bbox(ann["bbox"], w, h)
                    f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    # val 변환
    for img in tqdm(val_images):
        img_name = img["file_name"]
        img_path = os.path.join(train_image_path, img_name)

        shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images/val", img_name))

        if img["id"] in annotation_by_image_id:
            w, h = img["width"], img["height"]
            with open(os.path.join(OUTPUT_DIR, "labels/val", img_name.replace(os.path.splitext(img_name)[1], ".txt")), "w") as f:
                for ann in annotation_by_image_id[img["id"]]:
                    cls_id = category_mapping[ann["category_id"]]
                    bbox = convert_bbox(ann["bbox"], w, h)
                    f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    

    # data.yaml 생성
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("names:\n")
        for k, v in category_mapping.items():
            f.write(f"  {v}: class_{k}\n")

    print(f"[완료] 변환 완료 → {OUTPUT_DIR}")
    print(f"[생성] data.yaml → {yaml_path}")
    print("\nYOLO 학습 명령 예시:")
    print(f"yolo train model=yolov8l.pt data={yaml_path} epochs=50 imgsz=640")
    
    return category_reverse_mapping


if __name__ == "__main__":
    print("=== 프로그램 시작 ===")
    category_reverse_mapping = main()
    print("=== 프로그램 종료 ===")


# YOLO 학습 돌려봄
# YOLO_MODEL = "yolov8l.pt"


