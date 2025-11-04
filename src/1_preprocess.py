import json
import os
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from collections import defaultdict

# --- 0. 설정 ---
VALIDATION_SPLIT_RATIO = 0.2  # 20%를 검증용으로 사용
RANDOM_SEED = 42  # ⬅️ [추가] 재현성을 위한 랜덤 시드 고정

# --- 1. 경로 설정 ---
# (경로를 현재 폴더 기준으로 재수정)
base_dir = Path('./preprocess_bf/') 
# (원본 데이터 폴더명은 가이드에 맞게 수정)
annotation_root_dir = base_dir / 'train_annotations' 
image_root_dir = base_dir / 'train_images'

# (출력 폴더명은 가이드에 맞게 수정)

output_base_dir = Path('./preprocess_af/') 
train_image_dir = output_base_dir / 'images' / 'train'
val_image_dir = output_base_dir / 'images' / 'val'
train_label_dir = output_base_dir / 'labels' / 'train'
val_label_dir = output_base_dir / 'labels' / 'val'

# 기존 dataset 폴더가 있다면 삭제
if output_base_dir.exists():
    print(f"기존 '{output_base_dir}' 폴더를 삭제합니다...")
    shutil.rmtree(output_base_dir)

# 모든 새 폴더 생성
print(f"'{output_base_dir}' 폴더 구조를 생성합니다...")
train_image_dir.mkdir(parents=True, exist_ok=True)
val_image_dir.mkdir(parents=True, exist_ok=True)
train_label_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)


# --- 2. [1단계] 전체 클래스 맵 생성 ---
print("\n1단계: 전체 클래스 맵 생성 시작...")
json_paths = list(annotation_root_dir.glob('**/*.json')) 
if not json_paths:
    print(f" [오류] '{annotation_root_dir}' 폴더에서 .json 파일을 찾을 수 없습니다.")
    exit()

original_id_to_name = {}
for json_path in tqdm(json_paths, desc="클래스 수집 중"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for category in data.get('categories', []):
            cat_id = category['id']
            cat_name = category['name']
            if cat_id not in original_id_to_name:
                original_id_to_name[cat_id] = cat_name
    except Exception as e:
        print(f"파일 읽기 오류 {json_path}: {e}")

sorted_original_ids = sorted(original_id_to_name.keys())
original_id_to_yolo_id = {original_id: yolo_id for yolo_id, original_id in enumerate(sorted_original_ids)}

if not original_id_to_yolo_id:
    print(" [오류] 'categories' 정보를 가진 클래스를 찾지 못했습니다.")
    exit()
print(f"총 {len(original_id_to_yolo_id)}개의 고유 클래스를 찾았습니다.")


# --- 3. [2단계] 이미지별로 모든 Annotation 그룹화 ---
print("\n2단계: 유효한 이미지 기준으로 모든 라벨 그룹화 중...")

image_data_map = defaultdict(lambda: {'width': 0, 'height': 0, 'annotations': []})
missing_images = 0
json_processed = 0

for json_path in tqdm(json_paths, desc="라벨 그룹화 중"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_info = data['images'][0]
        file_name = image_info['file_name']
        source_image_path = image_root_dir / file_name 
        
        if source_image_path.exists():
            image_data_map[file_name]['width'] = image_info['width']
            image_data_map[file_name]['height'] = image_info['height']
            
            for ann in data.get('annotations', []):
                image_data_map[file_name]['annotations'].append(ann)
            json_processed += 1
        else:
            missing_images += 1
            
    except Exception as e:
        print(f"JSON 파일 처리 오류 {json_path}: {e}")

print(f"총 {len(image_data_map)}개의 고유한 이미지를 기준으로 {json_processed}개의 라벨을 그룹화했습니다.")
if missing_images > 0:
    print(f"(참고: {missing_images}개의 라벨에 해당하는 이미지가 없어 제외되었습니다.)")


# --- 4. [3단계] 고유 이미지 리스트를 8:2로 분할 및 변환 ---
print(f"\n3단계: 고유 이미지 {len(image_data_map)}개를 8:2 비율로 분할 시작...")

unique_image_filenames = list(image_data_map.keys())

# [핵심 수정] 
# 1. 시드 고정
print(f"RANDOM_SEED={RANDOM_SEED} 기준으로 데이터를 섞습니다.")
random.seed(RANDOM_SEED)

# 2. 리스트 섞기 (이제 항상 동일한 순서로 섞임)
random.shuffle(unique_image_filenames)

split_index = int(len(unique_image_filenames) * (1 - VALIDATION_SPLIT_RATIO))

train_files = unique_image_filenames[:split_index]
val_files = unique_image_filenames[split_index:]

print(f"  훈련용 이미지: {len(train_files)}개, 검증용 이미지: {len(val_files)}개")

conversion_errors = 0

for split_name, file_list, dest_img_dir, dest_lab_dir in [
    ('Train', train_files, train_image_dir, train_label_dir),
    ('Val', val_files, val_image_dir, val_label_dir)
]:
    print(f"\n{split_name} 세트 처리 중...")
    for file_name in tqdm(file_list, desc=f"{split_name} 변환 중"):
        try:
            img_data = image_data_map[file_name]
            img_width = img_data['width']
            img_height = img_data['height']
            
            label_file_name = Path(file_name).stem + '.txt'
            label_output_path = dest_lab_dir / label_file_name
            
            yolo_labels = []
            for ann in img_data['annotations']:
                original_cat_id = ann['category_id']
                yolo_class_id = original_id_to_yolo_id[original_cat_id]
                
                bbox = ann['bbox'] # [xmin, ymin, w, h]
                xmin, ymin, w, h = bbox
                
                x_center_norm = (xmin + w / 2) / img_width
                y_center_norm = (ymin + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                yolo_labels.append(f"{yolo_class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}")
            
            with open(label_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_labels))

            source_image_path = image_root_dir / file_name
            dest_image_path = dest_img_dir / file_name
            shutil.copyfile(source_image_path, dest_image_path)
            
        except Exception as e:
            print(f"파일 변환 오류 {file_name}: {e}")
            conversion_errors += 1

# --- 5. 최종 결과 확인 ---
print("\n--- 최종 분할 완료! ---")
train_img_count = len(list(train_image_dir.glob('*.png')))
val_img_count = len(list(val_image_dir.glob('*.png')))
total_img_count = train_img_count + val_img_count

if total_img_count > 0:
    print(f"  훈련용 이미지: {train_img_count}개 ({train_img_count/total_img_count*100:.1f}%)")
    print(f"  검증용 이미지: {val_img_count}개 ({val_img_count/total_img_count*100:.1f}%)")
    print(f"  훈련용 라벨: {len(list(train_label_dir.glob('*.txt')))}개")
    print(f"  검증용 라벨: {len(list(val_label_dir.glob('*.txt')))}개")
else:
    print("  처리된 이미지가 없습니다.")

if conversion_errors > 0:
    print(f"총 {conversion_errors}개의 파일 변환 중 오류가 발생했습니다.")