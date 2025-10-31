import glob
import os
import csv
import yaml
import json
from tqdm import tqdm
from pathlib import Path
import re
from ultralytics import YOLO

# --- [필수] 사용자 설정 ---
BASE_PROJECT_PATH = '.' # AI_Team2 폴더
PREPROCESSED_DIR = os.path.join(BASE_PROJECT_PATH, 'preprocessed_data')
ORIGINAL_DIR = os.path.join(BASE_PROJECT_PATH, 'ORIGINAL')

CATEGORY_MAP_PATH = os.path.join(PREPROCESSED_DIR, 'kaggle_category_map.json')

def get_latest_run_directory(base_dir="./runs_local/"):
    """
    base_dir(기본 검색 경로) 안에서 'val', 'predict' 등을 제외한
    실제 *학습* 폴더 중 가장 최근의 폴더 경로를 반환합니다.
    """
    all_dirs = glob.glob(os.path.join(base_dir, '*/'))
    train_dirs = [
        d for d in all_dirs
        if 'val' not in os.path.basename(os.path.normpath(d)) and
           'predict' not in os.path.basename(os.path.normpath(d))
    ]
    if not train_dirs:
        print(f"오류: '{base_dir}' 경로에서 'val' 또는 'predict'가 아닌 학습 폴더를 찾을 수 없습니다.")
        return None
    latest_dir = max(train_dirs, key=os.path.getmtime)
    print(f"가장 최신 *학습* 폴더를 찾았습니다: {latest_dir}")
    return latest_dir

def load_category_map(map_path):
    """ JSON 파일에서 {알약 이름: 캐글 ID} 맵 로드 """
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            name_to_kaggle_id = json.load(f)
        print(f"'{map_path}'에서 카테고리 맵 로드 성공 ({len(name_to_kaggle_id)}개).")
        return name_to_kaggle_id
    except Exception as e:
        print(f"🚨 오류: 카테고리 맵 파일 로드 실패. {e}")
        return None

def main():


# --- 1. Category ID 번역표 (총 73개로 수정됨) ---
    # CATEGORY_ID_MAP = [
    #     1899, 2482, 3350, 3482, 3543, 3742, 3831, 4377, 4542, 5093, 5885,
    #     6191, 6562, 10220, 12080, 12246, 12419, 12777, 13394, 13899,
    #     16231, 16261, 16547, 16550, 16687, 18109, 18146, 18356, 19231, 19551,
    #     19606, 19860, 20013, 20237, 20876, 21025, 21324, 21770, 22073, 22346,
    #     22361, 22626, 23202, 23222, 24849, 25366, 25437, 25468, 27652, 27732,
    #     27776, 27925, 27992, 28762, 29344, 29450, 29666, 29870, 30307, 31704,
    #     31862, 31884, 32309, 33008, 33207, 33877, 33879, 34596, 35205, 36636,
    #     38161, 41767, 44198
    # ]
    # --- 1. (!!!) 캐글 ID 매핑 로드 ---
    name_to_kaggle_id = load_category_map(CATEGORY_MAP_PATH)
    if name_to_kaggle_id is None:
        print("캐글 ID 맵 파일이 필요합니다. 1_create_category_map... 스크립트를 실행하세요.")
        return

    # --- 2. 최신 모델 로드 (원본 유지) ---
    latest_run_dir = get_latest_run_directory()
    if latest_run_dir is None:
        return

    trained_model_path = os.path.join(latest_run_dir, 'weights/best.pt')
    if not os.path.exists(trained_model_path):
        trained_model_path = os.path.join(latest_run_dir, 'weights/last.pt')
        if not os.path.exists(trained_model_path):
            print(f"오류: {latest_run_dir}에서 'best.pt'와 'last.pt'를 모두 찾을 수 없습니다.")
            return
        print("'best.pt'를 못찾아서 'last.pt'를 대신 사용합니다.")

    print(f"모델 로드 중: {trained_model_path}")
    model = YOLO(trained_model_path)

    # --- 3. Test 이미지 경로 지정 (원본 유지) ---
    BASE_PROJECT_PATH = '.' # AI_Team2 폴더
    TEST_IMAGE_DIR = os.path.join(BASE_PROJECT_PATH, 'ORIGINAL', 'test_images')  # <-- [!!쓰시는 2조분들 경로 확인!!]

    # --- ★★★ (핵심 수정 1) 파일 목록 로드, 필터링, 정렬 ★★★ ---

    print(f"\n'{TEST_IMAGE_DIR}' 폴더에서 이미지 파일을 검색 및 정렬합니다...")

    # 1. 일단 모든 .png 파일을 찾습니다.
    all_png_files = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.png'))

    valid_image_list = []
    for image_path in all_png_files:
        filename = os.path.basename(image_path)

        # 2. (필터 1) macOS가 생성하는 '._' 숨김 파일을 건너뜁니다.
        if filename.startswith('._'):
            print(f"  (필터됨) 숨겨진 파일: {filename}")
            continue

        filename_stem = Path(filename).stem # (예: '1.png' -> '1')

        # 3. (필터 2) 파일 이름(확장자 제외)이 '숫자'로만 되어 있는지 확인합니다.
        if filename_stem.isdigit():
            image_id = int(filename_stem)
            # (image_id, image_path) 튜플로 리스트에 추가
            valid_image_list.append((image_id, image_path))
        else:
            # 'logo.png' (stem='logo') 같은 파일
            print(f"  (필터됨) 숫자가 아닌 파일: {filename}")
            continue

    # 4. 리스트를 image_id (튜플의 0번째 요소) 기준으로 숫자 오름차순 정렬합니다.
    valid_image_list.sort(key=lambda x: x[0])

    if not valid_image_list:
        print(f"오류: '{TEST_IMAGE_DIR}' 폴더에서 유효한 (예: 1.png, 2.png) 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(valid_image_list)}개의 유효한 이미지를 정렬했습니다. 예측을 시작합니다.")

    # --- ★★★ (핵심 수정 2) CSV 준비 및 순차 예측 루프 ★★★ ---

    output_csv_path = './submission_v4_perfect_800_001.csv' # <-- (파일 이름 변경)
    csv_header = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']

    annotation_id_counter = 1

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        # 5. 정렬된 'valid_image_list'를 순서대로 순회합니다.
        for image_id, image_path in valid_image_list:

            # (로그 추가) 현재 처리 중인 파일을 명확히 표시
            print(f"\n--- 💎 [ID: {image_id}] ({os.path.basename(image_path)}) 처리 중... ---")

            results = model.predict(source=image_path, device='cuda:0', conf=0.01, iou=0.5, max_det=4)
            boxes = results[0].boxes

            if boxes.shape[0] == 0:
                print("  -> 객체를 찾지 못했습니다.")
                continue

            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(cls_list)):

                model_output_index = cls_list[i]

                # if 0 <= model_output_index < len(CATEGORY_ID_MAP):
                #     submission_category_id = CATEGORY_ID_MAP[model_output_index]
                # else:
                #     print(f"  -> (오류) 맵에 없는 클래스 인덱스({model_output_index})가 감지되어 건너뜁니다.")
                #     continue

                # (!!!) [수정된 매핑 1단계] 모델 내부 ID -> 클래스 이름
                # 예: 5 -> '아빌리파이정 10mg'
                if model_output_index not in model.names: # model.names는 {0: '이름1', 1: '이름2'...} 딕셔너리
                    tqdm.write(f"  -> (오류) 모델에 없는 클래스 ID({model_output_index}) 감지.")
                    continue
                class_name = model.names[model_output_index]

                # (!!!) [수정된 매핑 2단계] 클래스 이름 -> 캐글 Category ID
                # 예: '아빌리파이정 10mg' -> 3543 (name_to_kaggle_id 딕셔너리 사용)
                if class_name not in name_to_kaggle_id:
                    tqdm.write(f"  -> (오류) 캐글 맵('{CATEGORY_MAP_PATH}')에 없는 클래스 이름('{class_name}') 감지.")
                    continue
                submission_category_id = name_to_kaggle_id[class_name]

                score = conf_list[i]

                x1, y1, x2, y2 = xyxy_list[i]
                bbox_x = x1
                bbox_y = y1
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                # (로그 추가) 발견된 객체 정보를 CSV에 쓰기 *직전*에 터미널에 출력
                print(f"  -> ✅ 발견: Category={submission_category_id}, Score={score:.4f}, BBox=[{bbox_x:.1f}, {bbox_y:.1f}, {bbox_w:.1f}, {bbox_h:.1f}]")

                writer.writerow([
                    annotation_id_counter,
                    image_id,
                    submission_category_id,
                    round(bbox_x, 4),
                    round(bbox_y, 4),
                    round(bbox_w, 4),
                    round(bbox_h, 4),
                    round(score, 4)
                ])
                annotation_id_counter += 1

    print(f"\n\n🎉 예측 완료!")
    print(f"새로운 *정렬된* 제출용 CSV 파일이 다음 경로에 저장되었습니다: {output_csv_path}")

if __name__ == '__main__':
    main()