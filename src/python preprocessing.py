import os
import glob
import yaml # PyYAML 설치 필요: pip install pyyaml

# --- 설정값 ---
# 이 스크립트는 'src' 폴더 내에서 실행되어야 합니다.

# 1. 프로젝트 루트 경로 ('src' 폴더의 부모 폴더, 즉 'AI_TEAM2')
PROJECT_ROOT = os.path.dirname(os.getcwd())

# 2. YOLO 데이터셋 경로 (AI_TEAM2/images, AI_TEAM2/labels)
IMG_DIR = os.path.join(PROJECT_ROOT, "images")
LBL_DIR = os.path.join(PROJECT_ROOT, "labels")

# 3. 생성될 data.yaml 파일 경로 (AI_TEAM2/data.yaml)
OUTPUT_YAML_PATH = os.path.join(PROJECT_ROOT, "data.yaml")
# ----------------

def find_max_class_index(label_folders):
    """주어진 라벨 폴더들에서 가장 큰 클래스 인덱스를 찾습니다."""
    max_index = -1
    files_scanned = 0
    
    print("라벨 파일 스캔 중 (클래스 개수 파악)...")
    for folder in label_folders:
        if not os.path.exists(folder):
            print(f"[경고] 라벨 폴더 없음: {folder}")
            continue
            
        # glob.glob으로 폴더 내 모든 .txt 파일 경로 가져오기
        label_files = glob.glob(os.path.join(folder, '*.txt'))
        files_scanned += len(label_files)
        
        for file_path in label_files:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            class_index = int(parts[0])
                            if class_index > max_index:
                                max_index = class_index
            except Exception as e:
                print(f"[오류] 파일 읽기 실패 {file_path}: {e}")
                
    print(f"총 {files_scanned}개의 라벨 파일 스캔 완료.")
    return max_index

def create_data_yaml():
    """기존 데이터셋 구조를 바탕으로 data.yaml 파일을 생성합니다."""
    
    train_img_path = os.path.join(IMG_DIR, "train")
    val_img_path = os.path.join(IMG_DIR, "val")
    train_lbl_path = os.path.join(LBL_DIR, "train")
    val_lbl_path = os.path.join(LBL_DIR, "val")

    # 1. 필수 폴더 존재 확인
    if not os.path.exists(train_img_path):
        print(f"[오류] Train 이미지 폴더 없음: {train_img_path}")
        return
    if not os.path.exists(val_img_path):
         print(f"[오류] Val 이미지 폴더 없음: {val_img_path}")
         return
    if not os.path.exists(train_lbl_path):
         print(f"[경고] Train 라벨 폴더 없음: {train_lbl_path} (클래스 개수 파악에 영향)")
    if not os.path.exists(val_lbl_path):
         print(f"[경고] Val 라벨 폴더 없음: {val_lbl_path} (클래스 개수 파악에 영향)")

    # 2. 라벨 파일 스캔하여 클래스 개수(nc) 결정
    max_class_index = find_max_class_index([train_lbl_path, val_lbl_path])
    
    if max_class_index == -1:
        print("[오류] 라벨 파일에서 클래스 인덱스를 찾을 수 없습니다. 라벨 파일 형식을 확인하세요.")
        print("       (형식 예: 0 0.5 0.5 0.2 0.2)")
        nc = 0 # 클래스를 찾지 못함
    else:
        nc = max_class_index + 1 # 인덱스는 0부터 시작하므로 +1
        print(f"가장 큰 클래스 인덱스: {max_class_index} -> 총 클래스 개수(nc): {nc}")

    # 3. 임시 클래스 이름 생성 (⚠️ 반드시 수동으로 수정 필요!)
    class_names = [f'class_{i}' for i in range(nc)]

    # 4. YAML 데이터 구조 생성
    data = {
        'path': PROJECT_ROOT,  # 데이터셋 루트 경로 (AI_TEAM2 폴더)
        'train': os.path.relpath(train_img_path, PROJECT_ROOT), # 루트 기준 상대경로 (images/train)
        'val': os.path.relpath(val_img_path, PROJECT_ROOT),     # 루트 기준 상대경로 (images/val)
        'nc': nc,
        'names': class_names
    }

    # 5. YAML 파일 쓰기
    try:
        with open(OUTPUT_YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"[성공] data.yaml 파일 생성 완료: {OUTPUT_YAML_PATH}")
        print("\n" + "="*30)
        print("⚠️ 중요: 생성된 data.yaml 파일을 열어서")
        print("   'names:' 항목 아래의 클래스 이름들을")
        print("   실제 알약 이름으로 *반드시* 수정해주세요!")
        print("="*30)
        
    except Exception as e:
        print(f"[오류] data.yaml 파일 쓰기 실패: {e}")


if __name__ == "__main__":
    print("=== data.yaml 생성 프로그램 시작 ===")
    create_data_yaml()
    print("=== data.yaml 생성 프로그램 종료 ===")