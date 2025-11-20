import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# 설정
NPZ_FILE = 'data_L5_R5_nyse.npz'
OUTPUT_FOLDER = 'validation_samples_nyse'
NUM_SAMPLES = 100
NUM_PER_LABEL = NUM_SAMPLES // 2  # Up 50개, Down 50개

# 출력 폴더 생성
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"'{NPZ_FILE}' 파일 로드 중...")
data = np.load(NPZ_FILE, mmap_mode='r')

images = data['images']
labels = data['labels']
dates = data['dates']
returns = data['returns']
tickers = data['tickers']

total_count = len(images)
print(f"총 {total_count:,}개의 샘플 발견.")
print(f"라벨 분포: Up={np.sum(labels==1):,}, Down={np.sum(labels==0):,}")

# 라벨별 인덱스 분리
up_indices = np.where(labels == 1)[0]
down_indices = np.where(labels == 0)[0]

print(f"\nUp 샘플: {len(up_indices):,}개")
print(f"Down 샘플: {len(down_indices):,}개")

def select_diverse_samples(indices, returns_subset, num_samples, label_name):
    """
    다양한 수익률 분포를 가진 샘플을 선택합니다.
    - 극단값 (상위/하위 5%)
    - 중간값 (25%, 50%, 75% 분위수)
    - 0에 가까운 값
    """
    returns_values = returns_subset
    
    # 이상치 제거를 위해 IQR 방법 사용
    q1 = np.percentile(returns_values, 25)
    q3 = np.percentile(returns_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    # 정상 범위 내의 인덱스
    normal_mask = (returns_values >= lower_bound) & (returns_values <= upper_bound)
    normal_indices = indices[normal_mask]
    normal_returns = returns_values[normal_mask]
    
    selected_indices = []
    
    # 1. 극단값 (상위/하위 5%) - 각각 10개
    if len(normal_returns) > 0:
        top_5_percent = int(len(normal_returns) * 0.95)
        bottom_5_percent = int(len(normal_returns) * 0.05)
        
        top_indices = normal_indices[np.argsort(normal_returns)[-top_5_percent:]]
        bottom_indices = normal_indices[np.argsort(normal_returns)[:bottom_5_percent]]
        
        # 극단값에서 균등하게 선택
        if len(top_indices) > 0:
            top_selected = np.linspace(0, len(top_indices)-1, min(10, len(top_indices)), dtype=int)
            selected_indices.extend(top_indices[top_selected])
        
        if len(bottom_indices) > 0:
            bottom_selected = np.linspace(0, len(bottom_indices)-1, min(10, len(bottom_indices)), dtype=int)
            selected_indices.extend(bottom_indices[bottom_selected])
    
    # 2. 분위수 기반 선택 (25%, 50%, 75%) - 각각 5개씩
    if len(normal_returns) > 0:
        percentiles = [25, 50, 75]
        for p in percentiles:
            p_value = np.percentile(normal_returns, p)
            p_indices = normal_indices[np.abs(normal_returns - p_value) < 0.01]
            if len(p_indices) > 0:
                selected = np.random.choice(p_indices, min(5, len(p_indices)), replace=False)
                selected_indices.extend(selected)
    
    # 3. 0에 가까운 값 (수익률이 거의 0인 경우) - 5개
    if len(normal_returns) > 0:
        near_zero_indices = normal_indices[np.abs(normal_returns) < 0.01]
        if len(near_zero_indices) > 0:
            selected = np.random.choice(near_zero_indices, min(5, len(near_zero_indices)), replace=False)
            selected_indices.extend(selected)
    
    # 4. 나머지는 랜덤 선택으로 채우기
    remaining = num_samples - len(selected_indices)
    if remaining > 0:
        remaining_indices = np.setdiff1d(indices, selected_indices)
        if len(remaining_indices) > 0:
            selected = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
            selected_indices.extend(selected)
    
    return np.array(selected_indices[:num_samples])

# Up 샘플 선택
print(f"\nUp 샘플 {NUM_PER_LABEL}개 선택 중...")
up_returns = returns[up_indices]
up_selected = select_diverse_samples(up_indices, up_returns, NUM_PER_LABEL, "Up")

# Down 샘플 선택
print(f"Down 샘플 {NUM_PER_LABEL}개 선택 중...")
down_returns = returns[down_indices]
down_selected = select_diverse_samples(down_indices, down_returns, NUM_PER_LABEL, "Down")

# 최종 선택된 인덱스
all_selected = np.concatenate([up_selected, down_selected])
np.random.shuffle(all_selected)  # 순서 섞기

print(f"\n총 {len(all_selected)}개의 샘플 선택 완료.")
print(f"  - Up: {len(up_selected)}개")
print(f"  - Down: {len(down_selected)}개")

# 선택된 샘플의 통계 정보
selected_returns = returns[all_selected]
selected_labels = labels[all_selected]
selected_tickers = tickers[all_selected]

print(f"\n선택된 샘플 통계:")
print(f"  - 수익률 범위: {np.min(selected_returns):.4f} ~ {np.max(selected_returns):.4f}")
print(f"  - 수익률 평균: {np.mean(selected_returns):.4f}")
print(f"  - 고유 티커 수: {len(np.unique(selected_tickers))}개")

# 이미지 저장 및 메타데이터 저장
metadata = []
print(f"\n이미지 저장 중...")

for i, idx in enumerate(all_selected):
    sample_image = images[idx]
    sample_label = labels[idx]
    sample_date = dates[idx]
    sample_return = returns[idx]
    sample_ticker = tickers[idx]
    
    # 날짜 포맷팅
    if isinstance(sample_date, np.datetime64):
        date_str = np.datetime_as_string(sample_date, unit='D')
    elif isinstance(sample_date, str):
        date_str = sample_date
    else:
        date_str = str(sample_date)
    
    # 파일명 생성 (인덱스_티커_날짜_라벨_수익률.png)
    filename = f"{i:03d}_{sample_ticker}_{date_str}_L{sample_label}_R{sample_return:.4f}.png"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    # 이미지 저장
    img_to_save = sample_image.squeeze().copy()
    h, w = img_to_save.shape
    price_height = int(h * 4 / 5)  # 가격 영역 높이
    
    # 이미지를 RGB로 변환하여 이동평균선을 회색으로 표시
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 모든 픽셀을 흑백으로 복사
    img_rgb[:, :, 0] = img_to_save
    img_rgb[:, :, 1] = img_to_save
    img_rgb[:, :, 2] = img_to_save
    
    # 이동평균선을 회색으로 표시
    # 이동평균선은 x_center (1, 4, 7, 10, 13) 위치에 단일 픽셀로 그려짐
    # High-Low 바와 구분하기 위해, 각 x_center 위치에서 단일 픽셀인 경우를 이동평균선으로 간주
    for day in range(5):
        x_center = day * 3 + 1
        if x_center < w:
            # 가격 영역에서 x_center 열의 흰색 픽셀 확인
            price_col = img_to_save[:price_height, x_center]
            white_pixels = np.where(price_col == 255)[0]
            
            if len(white_pixels) > 0:
                # 연속된 픽셀 그룹 찾기 (High-Low 바는 연속된 픽셀)
                groups = []
                if len(white_pixels) > 0:
                    current_group = [white_pixels[0]]
                    for i in range(1, len(white_pixels)):
                        if white_pixels[i] == white_pixels[i-1] + 1:
                            current_group.append(white_pixels[i])
                        else:
                            groups.append(current_group)
                            current_group = [white_pixels[i]]
                    groups.append(current_group)
                
                # 단일 픽셀 그룹을 회색으로 변경 (이동평균선)
                for group in groups:
                    if len(group) == 1:
                        y_pos = group[0]
                        img_rgb[y_pos, x_center] = [128, 128, 128]  # 회색
    
    plt.figure(figsize=(6, 4))
    plt.imshow(img_rgb, interpolation='nearest')
    plt.title(f"Sample {i:03d}\nTicker: {sample_ticker}\nDate: {date_str}\nLabel: {sample_label} ({'UP' if sample_label == 1 else 'DOWN'})\nReturn: {sample_return:.4f} ({sample_return*100:.2f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    # 메타데이터 저장
    metadata.append({
        'index': i,
        'original_index': int(idx),
        'ticker': str(sample_ticker),
        'date': date_str,
        'label': int(sample_label),
        'return': float(sample_return),
        'filename': filename
    })
    
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(all_selected)} 저장 완료...")

# 메타데이터를 CSV로 저장
import pandas as pd
df_metadata = pd.DataFrame(metadata)
metadata_file = os.path.join(OUTPUT_FOLDER, 'metadata.csv')
df_metadata.to_csv(metadata_file, index=False)
print(f"\n메타데이터를 '{metadata_file}'에 저장했습니다.")

# 요약 정보 출력
print(f"\n=== 저장 완료 ===")
print(f"출력 폴더: {OUTPUT_FOLDER}")
print(f"총 이미지 수: {len(all_selected)}개")
print(f"메타데이터 파일: {metadata_file}")

data.close()
print("\n완료!")

