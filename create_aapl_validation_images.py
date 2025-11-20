import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# make_image.py에서 필요한 함수들을 직접 복사
def preprocess_data(df):
    """
    yfinance CSV를 논문에 맞게 전처리합니다.
    1. 날짜 인덱스 설정
    2. 조정 수익률(AdjReturn) 계산
    3. O/H/L 가격을 종가 대비 비율(factor)로 계산
    """
    # yfinance 데이터 형식에 맞춤
    if 'Date' not in df.columns:
        return None
        
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        return None
        
    df = df.set_index('Date').sort_index()

    # 1993-01-01 이전 데이터 제거
    df = df[df.index >= '1993-01-01']

    # 데이터가 너무 적으면 건너뜀
    if len(df) < 10:
        return None

    # 'Adj Close'를 사용해 조정 수익률(RET) 계산
    df['AdjReturn'] = df['Adj Close'].pct_change()
    
    # O, H, L 가격을 종가(Close) 대비 비율로 계산
    df['Close'] = df['Close'].replace(0, 1e-9)
    df['Open_factor'] = df['Open'] / df['Close']
    df['High_factor'] = df['High'] / df['Close']
    df['Low_factor'] = df['Low'] / df['Close']
    
    # 첫 번째 행은 수익률이 NaN이므로 제거
    df = df.iloc[1:]
    
    # 무한대/NaN 값 제거 (데이터 오류가 있을 경우)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def calculate_label_and_return(label_window_df):
    if label_window_df.empty or label_window_df['AdjReturn'].isnull().any():
        return None, None
    
    # 누적 수익률 계산 (예: 1.05 -> 5% 수익)
    cum_ret_factor = (1 + label_window_df['AdjReturn']).prod()
    
    # 0% (즉, 1.0) 보다 크면 1(Up), 아니면 0(Down)
    label = 1 if cum_ret_factor > 1.0 else 0
    
    # 실제 수익률 (예: 1.05 -> 0.05, 0.98 -> -0.02)
    actual_return = cum_ret_factor - 1.0
    
    return label, actual_return

# --- 설정 ---
AAPL_CSV = 'nasdaq_yfinance_20200401/stocks_sample/AAPL.csv'
OUTPUT_FOLDER = 'aapl_validation_images'
NUM_SAMPLES = 100
N_DAYS = 5

# 이미지 크기 설정 (원본과 동일: 15x32)
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 15  # 5일 * 3픽셀 = 15픽셀 (원본과 동일)

def generate_image_from_window_with_gray_ma(window_df, n_days, total_height, image_width):
    """
    make_image.py의 generate_image_from_window와 동일하지만,
    이동평균선을 회색(128)으로 표시합니다.
    """
    
    # 1. 상대 가격 시리즈 생성
    rel_prices = window_df.copy()
    
    # 첫 날 종가를 1로 정규화하기 위해 누적 수익률 계산
    rel_prices['RelClose'] = (1 + rel_prices['AdjReturn']).cumprod()
    
    # 이 윈도우의 첫날 RelClose가 1이 되도록 전체 윈도우를 스케일링
    first_rel_close = rel_prices['RelClose'].iloc[0]
    if first_rel_close == 0:
        return None  # 오류 방지
    
    rel_prices['RelClose'] = rel_prices['RelClose'] / first_rel_close
    
    # RelClose와 factor를 이용해 상대적인 O, H, L 가격 재구성
    rel_prices['RelOpen'] = rel_prices['Open_factor'] * rel_prices['RelClose']
    rel_prices['RelHigh'] = rel_prices['High_factor'] * rel_prices['RelClose']
    rel_prices['RelLow'] = rel_prices['Low_factor'] * rel_prices['RelClose']
    
    # 2. 이동평균 계산 (n일 이미지에 n일 이평선)
    rel_prices[f'MA'] = rel_prices['RelClose'].rolling(window=n_days, min_periods=1).mean()
    
    # 3. 스케일링 파라미터 찾기 (벡터화)
    price_cols = ['RelOpen', 'RelHigh', 'RelLow', 'RelClose', 'MA']
    all_prices = rel_prices[price_cols].values.flatten()
    
    min_price = np.nanmin(all_prices)
    max_price = np.nanmax(all_prices)
    
    # Check if min_price or max_price are NaN (can happen with all-NaN windows)
    if np.isnan(min_price) or np.isnan(max_price):
        return None

    max_volume = rel_prices['Volume'].max()
    
    # 4. 이미지(NumPy 배열) 생성
    image = np.zeros((total_height, image_width), dtype=np.uint8)
    
    # 가격과 거래량 영역 분리 (논문: 가격 4/5, 거래량 1/5)
    price_height = int(total_height * 4 / 5)
    volume_height = total_height - price_height

    # 스케일링 함수 (벡터화를 위해 numpy 배열을 받도록 수정)
    price_range = max_price - min_price
    if price_range == 0:
        price_range = 1.0  # 0으로 나누기 방지
    
    # 벡터화된 스케일링
    def scale_price_y_vec(prices):
        norm_prices = (prices - min_price) / price_range
        return ((price_height - 1) * (1 - norm_prices)).astype(np.int32)
    
    def scale_volume_h_vec(volumes):
        if max_volume == 0:
            return np.zeros_like(volumes, dtype=np.int32)
        return ((volumes / max_volume) * (volume_height - 1)).astype(np.int32)

    # 5. 벡터화된 픽셀 그리기 (하루에 3픽셀 너비)
    # 모든 가격을 한 번에 스케일링
    rel_high_vals = rel_prices['RelHigh'].values
    rel_low_vals = rel_prices['RelLow'].values
    rel_open_vals = rel_prices['RelOpen'].values
    rel_close_vals = rel_prices['RelClose'].values
    ma_vals = rel_prices['MA'].values
    volume_vals = rel_prices['Volume'].values
    
    y_high_arr = scale_price_y_vec(rel_high_vals)
    y_low_arr = scale_price_y_vec(rel_low_vals)
    y_open_arr = scale_price_y_vec(rel_open_vals)
    y_close_arr = scale_price_y_vec(rel_close_vals)
    y_ma_arr = scale_price_y_vec(ma_vals)
    vol_h_arr = scale_volume_h_vec(volume_vals)
    
    # x 좌표 배열 생성
    x_left_arr = np.arange(n_days) * 3
    x_center_arr = x_left_arr + 1
    x_right_arr = x_left_arr + 2
    
    # 벡터화된 픽셀 그리기
    for t in range(n_days):
        x_left, x_center, x_right = x_left_arr[t], x_center_arr[t], x_right_arr[t]
        
        # High-Low 바 (y_high부터 y_low까지)
        y_high, y_low = y_high_arr[t], y_low_arr[t]
        
        # Ensure y_high is less than or equal to y_low before slicing
        y1, y2 = min(y_high, y_low), max(y_high, y_low)
        image[y1:y2+1, x_center] = 255
        
        # Open, Close 점
        image[y_open_arr[t], x_left] = 255
        image[y_close_arr[t], x_right] = 255
        
        # 거래량 바 (픽셀을 좁게 - 1픽셀만)
        vol_h = vol_h_arr[t]
        if vol_h > 0:
            image[total_height - vol_h : total_height, x_center] = 255

    # 6. 이평선 그리기 (회색으로) - 원본과 다른 부분
    for t in range(n_days):
        if not np.isnan(y_ma_arr[t]):
            image[y_ma_arr[t], x_center_arr[t]] = 128  # 회색 점 (원본은 255)

    return image


def select_diverse_samples(df_processed, n_days, num_samples):
    """
    다양한 샘플을 선택합니다.
    - 시간적으로 골고루 분산
    - 라벨(Up/Down) 균형
    - 수익률 분포 다양성
    """
    min_length = n_days * 2  # 이미지 윈도우 + 라벨 윈도우
    
    if len(df_processed) < min_length:
        return []
    
    # 모든 가능한 샘플 생성
    all_samples = []
    for i in range(len(df_processed) - min_length + 1):
        img_window = df_processed.iloc[i : i + n_days]
        label_window = df_processed.iloc[i + n_days : i + n_days + n_days]
        
        label, actual_return = calculate_label_and_return(label_window)
        
        if label is not None and actual_return is not None:
            date = img_window.index[-1]
            all_samples.append({
                'index': i,
                'date': date,
                'label': label,
                'return': actual_return
            })
    
    if len(all_samples) == 0:
        return []
    
    # 라벨별로 분리
    up_samples = [s for s in all_samples if s['label'] == 1]
    down_samples = [s for s in all_samples if s['label'] == 0]
    
    selected = []
    
    # Up 샘플 선택 (50개)
    if len(up_samples) > 0:
        num_up = min(num_samples // 2, len(up_samples))
        # 수익률 분포를 고려하여 선택
        up_returns = np.array([s['return'] for s in up_samples])
        # 균등하게 분산된 인덱스 선택
        indices = np.linspace(0, len(up_samples) - 1, num_up, dtype=int)
        selected.extend([up_samples[i] for i in indices])
    
    # Down 샘플 선택 (50개)
    if len(down_samples) > 0:
        num_down = min(num_samples - len(selected), len(down_samples))
        # 수익률 분포를 고려하여 선택
        down_returns = np.array([s['return'] for s in down_samples])
        # 균등하게 분산된 인덱스 선택
        indices = np.linspace(0, len(down_samples) - 1, num_down, dtype=int)
        selected.extend([down_samples[i] for i in indices])
    
    # 나머지는 랜덤으로 채우기
    remaining = num_samples - len(selected)
    if remaining > 0:
        all_remaining = [s for s in all_samples if s not in selected]
        if len(all_remaining) > 0:
            np.random.seed(42)  # 재현성을 위해
            additional = np.random.choice(len(all_remaining), min(remaining, len(all_remaining)), replace=False)
            selected.extend([all_remaining[i] for i in additional])
    
    return selected[:num_samples]


def main():
    # 출력 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"출력 폴더: {OUTPUT_FOLDER}")
    
    # AAPL 데이터 로드 및 전처리
    print(f"\nAAPL 데이터 로드 중: {AAPL_CSV}")
    df = pd.read_csv(AAPL_CSV)
    df_processed = preprocess_data(df)
    
    if df_processed is None or len(df_processed) < N_DAYS * 2:
        print("오류: 데이터가 충분하지 않습니다.")
        return
    
    print(f"전처리 완료. 총 {len(df_processed)}일의 데이터")
    
    # 다양한 샘플 선택
    print(f"\n{NUM_SAMPLES}개의 샘플 선택 중...")
    selected_samples = select_diverse_samples(df_processed, N_DAYS, NUM_SAMPLES)
    
    if len(selected_samples) == 0:
        print("오류: 선택된 샘플이 없습니다.")
        return
    
    print(f"총 {len(selected_samples)}개의 샘플 선택 완료")
    print(f"  - Up: {sum(1 for s in selected_samples if s['label'] == 1)}개")
    print(f"  - Down: {sum(1 for s in selected_samples if s['label'] == 0)}개")
    
    # 이미지 생성 및 저장
    print(f"\n이미지 생성 및 저장 중...")
    metadata = []
    
    for i, sample in enumerate(selected_samples):
        idx = sample['index']
        
        # 이미지 윈도우 추출
        img_window = df_processed.iloc[idx : idx + N_DAYS]
        
        # 이미지 생성 (원본 크기 15x32 사용, 이동평균선만 회색)
        image = generate_image_from_window_with_gray_ma(
            img_window, N_DAYS, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        
        if image is None:
            continue
        
        # 날짜 포맷팅
        date_str = sample['date'].strftime('%Y-%m-%d') if hasattr(sample['date'], 'strftime') else str(sample['date'])
        
        # 파일명 생성
        filename = f"{i:03d}_AAPL_{date_str}_L{sample['label']}_R{sample['return']:.4f}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        # 이미지 저장 (sample_73.png 스타일로)
        # (32, 15) 형태의 흑백 이미지를 cmap='gray'로 표시, aspect='auto'로 가로로 넓게
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(image, cmap='gray', interpolation='nearest', aspect='auto')
        ax.set_title(
            f"Sample Index: {i} | Ticker: AAPL | Date: {date_str} | Label: {sample['label']} ({'UP' if sample['label'] == 1 else 'DOWN'}) | Return: {sample['return']:.4f} ({sample['return']*100:.2f}%)",
            fontsize=9
        )
        ax.set_xlabel("Features (Time steps)")
        ax.set_ylabel("Channels (LOB data)")
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 메타데이터 저장
        metadata.append({
            'index': i,
            'original_index': idx,
            'ticker': 'AAPL',
            'date': date_str,
            'label': sample['label'],
            'return': sample['return'],
            'filename': filename
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(selected_samples)} 저장 완료...")
    
    # 메타데이터를 CSV로 저장
    df_metadata = pd.DataFrame(metadata)
    metadata_file = os.path.join(OUTPUT_FOLDER, 'metadata.csv')
    df_metadata.to_csv(metadata_file, index=False)
    
    print(f"\n=== 저장 완료 ===")
    print(f"출력 폴더: {OUTPUT_FOLDER}")
    print(f"총 이미지 수: {len(metadata)}개")
    print(f"메타데이터 파일: {metadata_file}")
    print(f"이미지 크기: {IMAGE_HEIGHT}x{IMAGE_WIDTH} (원본과 동일, 시각화만 가로 확장)")
    print(f"이동평균선: 회색(128)으로 표시")


if __name__ == "__main__":
    main()

