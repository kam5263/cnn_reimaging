import pandas as pd
import numpy as np
import cv2  # OpenCV
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# --- 1. 상수 정의 (논문 기반) ---

# 논문에서 언급된 이미지 크기 정의
IMAGE_DIMS = {
    5: {'h': 32, 'w': 15},  # 5일 (3px * 5일 = 15px 너비)
    20: {'h': 64, 'w': 60}, # 20일 (3px * 20일 = 60px 너비)
    60: {'h': 96, 'w': 180} # 60일 (3px * 60일 = 180px 너비)
}

# --- 2. 헬퍼 함수 (데이터 전처리 및 이미지 생성) ---

def preprocess_data(df):
    """
    yfinance CSV를 논문에 맞게 전처리합니다.
    1. 날짜 인덱스 설정
    2. 조정 수익률(AdjReturn) 계산
    3. O/H/L 가격을 종가 대비 비율(factor)로 계산
    """
    # yfinance 데이터 형식에 맞춤
    if 'Date' not in df.columns:
        raise ValueError("'Date' 컬럼이 CSV 파일에 없습니다.")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # 1993-01-01 이전 데이터 제거
    df = df[df.index >= '1993-01-01']

    # 데이터가 너무 적으면 건너뜀
    if len(df) < 10:
        return None

    # 'Adj Close'를 사용해 조정 수익률(RET) 계산
    df['AdjReturn'] = df['Adj Close'].pct_change()
    
    # O, H, L 가격을 종가(Close) 대비 비율로 계산
    # (나중에 상대 가격 시리즈를 재구성할 때 사용)
    # 0으로 나누는 것을 방지하기 위해 0인 종가는 아주 작은 값으로 대체
    df['Close'] = df['Close'].replace(0, 1e-9)
    df['Open_factor'] = df['Open'] / df['Close']
    df['High_factor'] = df['High'] / df['Close']
    df['Low_factor'] = df['Low'] / df['Close']
    
    # 첫 번째 행은 수익률이 NaN이므로 제거
    df = df.iloc[1:]
    
    # 무한대/NaN 값 제거 (데이터 오류가 있을 경우)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def generate_image_from_window(window_df, n_days, total_height, image_width):
    """
    주어진 데이터 윈도우(n_days)로 하나의 차트 이미지를 생성합니다.
    논문의 핵심 스케일링 로직을 구현합니다.
    벡터화된 버전으로 최적화되었습니다.
    """
    
    # 1. 상대 가격 시리즈 생성
    rel_prices = window_df.copy()
    
    # 첫 날 종가를 1로 정규화하기 위해 누적 수익률 계산
    rel_prices['RelClose'] = (1 + rel_prices['AdjReturn']).cumprod()
    
    # 이 윈도우의 첫날 RelClose가 1이 되도록 전체 윈도우를 스케일링
    first_rel_close = rel_prices['RelClose'].iloc[0]
    if first_rel_close == 0:
        return None # 오류 방지
    
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
        if y_high <= y_low:
            image[y_high:y_low+1, x_center] = 255
        
        # Open, Close 점
        image[y_open_arr[t], x_left] = 255
        image[y_close_arr[t], x_right] = 255
        
        # 거래량 바
        vol_h = vol_h_arr[t]
        if vol_h > 0:
            image[total_height - vol_h : total_height, x_left : x_right + 1] = 255

    # 6. 이평선 그리기 (cv2.polylines로 점들 연결)
    ma_points = np.column_stack((x_center_arr, y_ma_arr))
    pts = ma_points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=False, color=255, thickness=1)

    return image

# 라벨과 "실제 수익률"을 함께 반환
def calculate_label_and_return(label_window_df):
    if label_window_df.empty:
        return None, None
    
    # 누적 수익률 계산 (예: 1.05 -> 5% 수익)
    cum_ret_factor = (1 + label_window_df['AdjReturn']).prod()
    
    # 0% (즉, 1.0) 보다 크면 1(Up), 아니면 0(Down)
    label = 1 if cum_ret_factor > 1.0 else 0
    
    # 실제 수익률 (예: 1.05 -> 0.05, 0.98 -> -0.02)
    actual_return = cum_ret_factor - 1.0
    
    return label, actual_return

def calculate_label(label_window_df):
    """
    미래 윈도우의 수익률을 기반으로 0 또는 1 라벨을 계산합니다.
    """
    if label_window_df.empty:
        return None
    
    # 누적 수익률 계산
    cum_ret = (1 + label_window_df['AdjReturn']).prod()
    
    # 0% (즉, 1.0) 보다 크면 1(Up), 아니면 0(Down)
    return 1 if cum_ret > 1.0 else 0

def process_single_file(filepath, n_days, img_config, min_length):
    """
    단일 CSV 파일을 처리하여 (이미지, 라벨, 날짜) 리스트를 반환합니다.
    멀티프로세싱을 위해 분리된 함수입니다.
    """
    results = []
    
    try:
        df = pd.read_csv(filepath)
        df_processed = preprocess_data(df)
        
        # 전처리 후 데이터가 너무 짧으면 건너뜀
        if df_processed is None or len(df_processed) < min_length:
            return results
            
        # 롤링 윈도우로 (이미지, 라벨) 쌍 생성
        for i in range(1, len(df_processed) - min_length):
            
            # 1. 이미지 데이터 추출 (i부터 i+n_days)
            img_window = df_processed.iloc[i : i + n_days]
            
            # 2. 라벨 데이터 추출 (그 다음 n_days)
            label_window = df_processed.iloc[i + n_days : i + n_days + n_days]
            
            # 3. 이미지 생성
            image = generate_image_from_window(
                img_window, n_days, img_config['h'], img_config['w']
            )
            
            # 4. 라벨 계산
            # label = calculate_label(label_window)
            label, actual_return = calculate_label_and_return(label_window)
            
            # 5. 날짜 저장 (이미지 윈도우의 마지막 날)
            date = img_window.index[-1]
            
            # if image is not None and label is not None:
            #     results.append((image, label, date))
            if image is not None and label is not None and actual_return is not None:
                # 실제 수익률(actual_return)도 결과에 추가
                results.append((image, label, date, actual_return))
    
    except pd.errors.EmptyDataError:
        pass  # 빈 파일은 조용히 건너뜀
    except Exception as e:
        pass  # 오류는 메인 프로세스에서 처리
    
    return results

# --- 3. 메인 파이프라인 함수 ---

def process_all_files(stocks_folder, output_file, n_days, num_workers=None):
    """
    지정된 폴더의 모든 CSV를 읽어 (이미지, 라벨, 날짜) 쌍을 생성하고
    하나의 .npz 파일로 저장합니다.
    멀티프로세싱을 사용하여 속도를 향상시킵니다.
    
    :param stocks_folder: CSV 파일들이 있는 폴더 경로 (예: 'stocks' 또는 '/stocks')
    :param output_file: 저장할 .npz 파일 이름 (예: 'data_60d.npz')
    :param n_days: 생성할 이미지 기간 (5, 20, 60 중 하나)
    :param num_workers: 사용할 프로세스 수 (None이면 CPU 코어 수 사용)
    """
    
    if n_days not in IMAGE_DIMS:
        raise ValueError(f"n_days는 {list(IMAGE_DIMS.keys())} 중 하나여야 합니다.")
        
    img_config = IMAGE_DIMS[n_days]
    
    # 이미지 윈도우(n_days)와 라벨 윈도우(n_days)에 필요한 최소 일수
    min_length = n_days + n_days
    
    # glob를 사용해 하위 폴더 포함 모든 csv 검색
    search_path = os.path.join(stocks_folder, "**", "*.csv")
    
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"경고: '{search_path}' 경로에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일 처리 시작 (n_days={n_days})...")
    
    # 멀티프로세싱 설정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 하나의 코어는 시스템용으로 남김
    
    print(f"멀티프로세싱 사용: {num_workers}개 프로세스")
    
    # 이미지(X), 정답(y), 날짜(meta)를 저장할 리스트
    all_images = []
    all_labels = []
    all_dates = []
    all_actual_returns = []
    # 멀티프로세싱으로 파일 처리
    process_func = partial(process_single_file, n_days=n_days, 
                          img_config=img_config, min_length=min_length)
    
    with Pool(processes=num_workers) as pool:
        # tqdm을 사용한 진행바와 함께 멀티프로세싱 실행
        results_list = list(tqdm(
            pool.imap(process_func, csv_files),
            total=len(csv_files),
            desc="CSV 파일 처리 중"
        ))
    
    # 결과 수집
    for results in results_list:
        for image, label, date, actual_return in results:
            all_images.append(image)
            all_labels.append(label)
            all_dates.append(date)
            all_actual_returns.append(actual_return)

    print("\n모든 파일 처리 완료. NumPy 배열로 변환 중...")
    
    # 리스트를 NumPy 배열로 변환
    images_arr = np.array(all_images)
    labels_arr = np.array(all_labels)
    dates_arr = np.array(all_dates)
    returns_arr = np.array(all_actual_returns)

    # 데이터 형태 출력
    print(f"총 {len(images_arr)}개의 (이미지, 라벨) 쌍이 생성되었습니다.")
    if len(images_arr) > 0:
        print(f"  이미지(X) 형태: {images_arr.shape}")
        print(f"  라벨(y) 형태: {labels_arr.shape}")
        print(f"  날짜(meta) 형태: {dates_arr.shape}")
        print(f"  실제 수익률(actual_return) 형태: {returns_arr.shape}")
        
        # 흑백 이미지이므로 채널 차원 추가 (H, W) -> (H, W, 1)
        # (TensorFlow/Keras 훈련에 적합하도록)
        images_arr = np.expand_dims(images_arr, axis=-1)
        print(f"  훈련용 이미지(X) 최종 형태: {images_arr.shape}")

        # np.savez_compressed: 여러 배열을 하나의 압축 파일로 저장
        print(f"데이터를 '{output_file}' 파일로 저장 중...")
        np.savez_compressed(
            output_file,
            images=images_arr,
            labels=labels_arr,
            dates=dates_arr,
            returns=returns_arr
        )
        print("저장 완료.")
    else:
        print("생성된 데이터가 없습니다. CSV 파일과 경로를 확인하세요.")


# --- 4. 메인 코드 실행 ---
if __name__ == "__main__":

    # 2. 파이프라인 실행
    
    # 60일 이미지 (L60)와 60일 후 라벨 (R60) 생성
    # process_all_files(
    #     stocks_folder=STOCKS_DIR, # 실제 경로(예: '/stocks')로 변경
    #     output_file='data_L60_R60.npz',
    #     n_days=60
    # )
    
    # 20일 이미지/라벨도 만들고 싶다면:
    # print("\n--- 3. 메인 파이프라인 실행 (20일 예제) ---")
    # process_all_files(
    #     stocks_folder=STOCKS_DIR, 
    #     output_file='data_L20_R20.npz',
    #     n_days=20
    # )
    
    # 5일 이미지/라벨도 만들고 싶다면:
    print("\n--- 4. 메인 파이프라인 실행 (5일 예제) ---")
    process_all_files(
        stocks_folder='nasdaq_yfinance_20200401/stocks', 
        output_file='data_L5_R5_with_returns.npz',
        n_days=5
    )