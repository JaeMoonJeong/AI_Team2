import json
from pathlib import Path
from tqdm import tqdm
import yaml # YAML 파일을 다루기 위해 추가

# --- 1. 경로 설정 ---

base_dir = Path('./').resolve() # (jypyter 등에서 실행 시) 현재 폴더의 '절대 경로'
annotation_root_dir = base_dir / 'preprocess_bf' # 'train_annotations' # 원본 JSON 폴더

dataset_dir = base_dir  / 'preprocess_af' # preprocess 스크립트가 생성한 폴더

# --- 2. [1단계] 클래스 정보 수집 ---
print(f"'{annotation_root_dir}' 폴더에서 클래스 맵을 생성합니다...")

json_paths = list(annotation_root_dir.glob('**/*.json')) 
if not json_paths:
    print(f" [오류] '{annotation_root_dir}' 폴더에서 .json 파일을 찾을 수 없습니다.")
    exit()

original_id_to_name = {} # {16547: "가바토파정 100mg", ...}
for json_path in tqdm(json_paths, desc="JSON 파일 읽는 중"):
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

if not original_id_to_name:
    print(" [오류] JSON 파일들을 읽었으나 'categories' 정보를 찾지 못했습니다.")
    exit()

# --- 3. [2단계] ID 매핑 및 'names' 리스트 생성 ---
# 원본 ID를 기준으로 '숫자 오름차순' 정렬
sorted_original_ids = sorted(original_id_to_name.keys())

# YOLO ID 순서대로 '이름' 리스트 생성
# 0번부터 순서대로 이름만 저장
yolo_names_list = [original_id_to_name[original_id] for original_id in sorted_original_ids]
nc = len(yolo_names_list) # 클래스 개수

print(f"\n총 {nc}개의 클래스를 찾았습니다.")
print("--- (화면 출력: 클래스 맵 확인용) ---")
print(f"{'YOLO ID':<10} | 이름")
print(f"{'-'*10:<10} | {'-'*30}")
for i, name in enumerate(yolo_names_list):
    print(f"{i:<10} | {name}")
print("---------------------------------")


# --- 4. [3단계] data.yaml 파일 생성 ---
print(f"\n'data.yaml' 파일 생성 중...")

# YAML 파일에 저장할 데이터 구성
yaml_data = {
    'path': str(dataset_dir), # dataset 폴더의 절대 경로
    'train': 'images/train',  # path 기준 상대 경로
    'val': 'images/val',      # path 기준 상대 경로
    
    'nc': nc,                 # 클래스 개수 (예: 73)
    'names': yolo_names_list  # 클래스 이름 리스트
}

# YAML 파일 저장 경로
yaml_file_path = base_dir / 'data.yaml'

try:
    # PyYAML 라이브러리가 필요합니다. (pip install PyYAML)
    import yaml
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        # allow_unicode=True: 한글이 깨지지 않게 저장
        # sort_keys=False: 딕셔너리 순서(path, train, val, nc, names)대로 저장
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        
    print(f"\n✅ 성공! '{yaml_file_path}' 파일이 생성되었습니다.")
    print("--- 생성된 파일 내용 미리보기 ---")
    print(f"path: {dataset_dir}")
    print("train: images/train")
    print("val: images/val")
    print(f"nc: {nc}")
    print(f"names: ['{yolo_names_list[0]}', '{yolo_names_list[1]}', ...]")
    
except ImportError:
    print("\n [오류] PyYAML 라이브러리가 필요합니다.")
    print(" 터미널에 'pip install PyYAML' 또는 'conda install PyYAML'을 입력하여 설치해주세요.")
except Exception as e:
    print(f"\n [오류] YAML 파일 저장 중 문제가 발생했습니다: {e}")