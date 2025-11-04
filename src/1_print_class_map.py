import json
from pathlib import Path
from tqdm import tqdm

# --- 1. 경로 설정 ---
# (경로를 현재 폴더 기준으로 재수정)

base_dir = Path('./preprocess_bf/') # jupyter 등에서 실행 시

# (경로 수정) 원본 JSON이 있는 'preprocess_bf' 폴더를 기준으로 함
annotation_root_dir = base_dir / 'train_annotations' 

if not annotation_root_dir.exists():
    print(f" [오류] 경로를 찾을 수 없습니다: {annotation_root_dir.resolve()}")
    print(" 'preprocess_bf/train_annotations' 폴더가 있는지 확인하세요.")
    exit()
    
# (경로 수정) 출력 파일도 'preprocess_bf' 폴더 안에 저장
output_dir = Path('./preprocess_af/')

# --- 2. [1단계] 클래스 정보 수집 ---
print(f"'{annotation_root_dir}' 폴더에서 클래스 맵을 생성합니다...")

# 모든 JSON 파일 경로 찾기
json_paths = list(annotation_root_dir.glob('**/*.json')) # 하위 폴더의 모든 .json 검색

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
    print(" [오류] JSON 파일들을 읽었으나 'categories' 정보를 가진 클래스를 찾지 못했습니다.")
    exit()

# --- 3. [2단계] ID 매핑 및 출력 ---
# 원본 ID를 기준으로 '숫자 오름차순' 정렬
sorted_original_ids = sorted(original_id_to_name.keys())

# [수정] 파일 저장을 위해, 출력할 내용을 리스트에도 저장
output_lines_for_txt = []

header1 = "\n" + "="*50
header2 = f"  [클래스 1:1 매칭 리스트] (총 {len(sorted_original_ids)}개)"
header3 = "="*50
header4 = f"{'YOLO ID':<10} | {'원본 Category ID':<20} | 이름"
header5 = f"{'-'*10:<10} | {'-'*20:<20} | {'-'*30}"

print(header1); output_lines_for_txt.append(header1.strip()) # 맨 앞 \n 제거
print(header2); output_lines_for_txt.append(header2)
print(header3); output_lines_for_txt.append(header3)
print(header4); output_lines_for_txt.append(header4)
print(header5); output_lines_for_txt.append(header5)

# {YOLO_ID: 원본_ID} 맵 생성 및 출력
yolo_id_to_original_id = {}
original_id_to_yolo_id = {}
yolo_id_to_name = {}

for yolo_id, original_id in enumerate(sorted_original_ids):
    # 3가지 버전의 맵을 모두 만듭니다.
    yolo_id_to_original_id[yolo_id] = original_id
    original_id_to_yolo_id[original_id] = yolo_id
    class_name = original_id_to_name[original_id]
    yolo_id_to_name[yolo_id] = class_name
    
    # 터미널에 출력
    line = f"{yolo_id:<10} | {original_id:<20} | {class_name}"
    print(line); output_lines_for_txt.append(line)

footer1 = "="*50
footer2 = "맵 생성 완료."
print(footer1); output_lines_for_txt.append(footer1)
print(footer2); output_lines_for_txt.append(footer2)


# --- [4단계] 파일로 저장 (수정된 부분) ---

# 1. 기계가 읽기 좋은 JSON 파일로 저장
# (submission.py 스크립트가 이 파일을 읽도록 만들 수 있습니다)
output_map_json_path = output_dir / 'class_map.json'
output_map_json = {
    'yolo_id_to_original_id': yolo_id_to_original_id, # {0: 1899, ...}
    'original_id_to_yolo_id': original_id_to_yolo_id, # {1899: 0, ...}
    'yolo_id_to_name': yolo_id_to_name               # {0: '보령부스파정 5mg', ...}
}
try:
    with open(output_map_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_map_json, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 기계용 맵 저장 완료: '{output_map_json_path.resolve()}'")
except Exception as e:
    print(f"\n [오류] JSON 파일 저장 실패: {e}")

# 2. 사람이 읽기 좋은 TXT 파일로 저장
# (터미널 출력을 그대로 복사한 파일)
output_map_txt_path = output_dir / 'class_map_readable.txt'
try:
    with open(output_map_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines_for_txt))
    print(f"✅ 사람용 맵 저장 완료: '{output_map_txt_path.resolve()}'")
except Exception as e:
    print(f"\n [오류] TXT 파일 저장 실패: {e}")
