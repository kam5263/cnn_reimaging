import pandas as pd
import numpy as np
import cv2  # OpenCV
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import platform # (디버깅용)

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
    NYSE/NASDAQ TXT 파일을 논문에 맞게 전처리합니다.
    1. 날짜 인덱스 설정
    2. 조정 수익률(AdjReturn) 계산 (이미 adjusted된 Close 사용)
    3. O/H/L 가격을 종가 대비 비율(factor)로 계산
    """
    # NYSE/NASDAQ 데이터 형식에 맞춤
    if 'Date' not in df.columns:
        # print("경고: 'Date' 컬럼이 파일에 없습니다.")
        return None
        
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        # print(f"경고: 날짜 변환 실패")
        return None
        
    df = df.set_index('Date').sort_index()

    # 1993-01-01 이전 데이터 제거
    df = df[df.index >= '1993-01-01']

    # 데이터가 너무 적으면 건너뜀
    if len(df) < 10:
        return None

    # 이미 adjusted된 'Close'를 사용해 조정 수익률(RET) 계산
    df['AdjReturn'] = df['Close'].pct_change()
    
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

    # 6. 이평선 그리기 (선으로 연결)
    # 연속된 점들을 선으로 연결
    for t in range(n_days - 1):
        if not np.isnan(y_ma_arr[t]) and not np.isnan(y_ma_arr[t + 1]):
            # 두 점 사이를 선으로 연결
            pt1 = (x_center_arr[t], y_ma_arr[t])
            pt2 = (x_center_arr[t + 1], y_ma_arr[t + 1])
            cv2.line(image, pt1, pt2, 255, 1)  # 흰색 선, 두께 1픽셀

    return image

# 라벨과 "실제 수익률"을 함께 반환
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


def process_single_file(filepath, n_days, img_config, min_length):
    """
    단일 TXT 파일을 처리하여 (이미지, 라벨, 날짜, 수익률, 티커) 리스트를 반환합니다.
    멀티프로세싱을 위해 분리된 함수입니다.
    """
    results = []
    
    # 파일 경로에서 티커 추출 (예: 'nyse_nasdaq_nyse_20171011/Stocks/aap.us.txt' -> 'aap.us')
    ticker = os.path.splitext(os.path.basename(filepath))[0]
    
    try:
        df = pd.read_csv(filepath)
        df_processed = preprocess_data(df)
        
        # 전처리 후 데이터가 너무 짧으면 건너뜀
        if df_processed is None or len(df_processed) < min_length:
            return results
            
        # 롤링 윈도우로 (이미지, 라벨) 쌍 생성
        for i in range(len(df_processed) - min_length + 1):
            
            # 1. 이미지 데이터 추출 (i부터 i+n_days)
            img_window = df_processed.iloc[i : i + n_days]
            
            # 2. 라벨 데이터 추출 (그 다음 n_days)
            label_window = df_processed.iloc[i + n_days : i + n_days + n_days]
            
            # 3. 이미지 생성
            image = generate_image_from_window(
                img_window, n_days, img_config['h'], img_config['w']
            )
            
            # 4. 라벨 계산
            label, actual_return = calculate_label_and_return(label_window)
            
            # 5. 날짜 저장 (이미지 윈도우의 마지막 날)
            date = img_window.index[-1]
            
            if image is not None and label is not None and actual_return is not None:
                # 실제 수익률(actual_return)과 티커 정보도 결과에 추가
                results.append((image, label, date, actual_return, ticker))
    
    except pd.errors.EmptyDataError:
        pass  # 빈 파일은 조용히 건너뜀
    except Exception as e:
        # print(f"파일 처리 중 오류: {filepath}, {e}")
        pass  # 오류는 메인 프로세스에서 처리
    
    return results

# --- 3. 메인 파이프라인 함수 ---

def process_all_files(stocks_folder, output_file, n_days, num_workers=None):
    """
    지정된 폴더의 모든 TXT 파일을 읽어 (이미지, 라벨, 날짜) 쌍을 생성하고
    하나의 .npz 파일로 저장합니다.
    멀티프로세싱을 사용하여 속도를 향상시킵니다.
    """
    
    if n_days not in IMAGE_DIMS:
        raise ValueError(f"n_days는 {list(IMAGE_DIMS.keys())} 중 하나여야 합니다.")
        
    img_config = IMAGE_DIMS[n_days]
    
    # 이미지 윈도우(n_days)와 라벨 윈도우(n_days)에 필요한 최소 일수
    min_length = n_days + n_days
    
    # glob를 사용해 하위 폴더 포함 모든 txt 파일 검색
    search_path = os.path.join(stocks_folder, "**", "*.txt")
    
    txt_files = glob.glob(search_path, recursive=True)
    
    if not txt_files:
        print(f"경고: '{search_path}' 경로에서 TXT 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(txt_files)}개의 TXT 파일 처리 시작 (n_days={n_days})...")
    
    # 멀티프로세싱 설정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 하나의 코어는 시스템용으로 남김
    
    print(f"멀티프로세싱 사용: {num_workers}개 프로세스")
    
    # 이미지(X), 정답(y), 날짜(meta)를 저장할 리스트
    all_images = []
    all_labels = []
    all_dates = []
    all_actual_returns = []
    all_tickers = []
    # 멀티프로세싱으로 파일 처리
    process_func = partial(process_single_file, n_days=n_days, 
                          img_config=img_config, min_length=min_length)
    
    with Pool(processes=num_workers) as pool:
        # tqdm을 사용한 진행바와 함께 멀티프로세싱 실행
        results_list = list(tqdm(
            pool.imap(process_func, txt_files),
            total=len(txt_files),
            desc="TXT 파일 처리 중"
        ))
    
    # 결과 수집
    print("\n멀티프로세싱 완료. 결과 수집 중...")
    for results in results_list:
        for image, label, date, actual_return, ticker in results:
            all_images.append(image)
            all_labels.append(label)
            
            # 🚨🚨🚨 여기가 핵심 수정 사항입니다! 🚨🚨🚨
            # Timestamp 객체 대신, 'YYYY-MM-DD' 형식의 문자열로 저장합니다.
            all_dates.append(date.strftime('%Y-%m-%d'))
            
            all_actual_returns.append(actual_return)
            all_tickers.append(ticker)

    print("모든 파일 처리 완료. NumPy 배열로 변환 중...")
    
    # 리스트를 NumPy 배열로 변환
    images_arr = np.array(all_images, dtype=np.uint8) # 메모리 효율을 위해 8-bit 정수
    labels_arr = np.array(all_labels, dtype=np.uint8) # 0 또는 1이므로 8-bit
    
    # 문자열로 변환되었으므로, np.array()는 dtype='<U10' (문자열) 배열을 생성합니다.
    # 이것은 'object'가 아니므로 pickle되지 않으며 mmap이 가능합니다.
    dates_arr = np.array(all_dates)
    
    returns_arr = np.array(all_actual_returns, dtype=np.float32) # 32-bit 부동소수점
    
    # 티커 정보도 문자열 배열로 저장
    tickers_arr = np.array(all_tickers)

    # 데이터 형태 출력
    print(f"총 {len(images_arr)}개의 (이미지, 라벨) 쌍이 생성되었습니다.")
    if len(images_arr) > 0:
        print(f"  이미지(X) 형태: {images_arr.shape}, dtype: {images_arr.dtype}")
        print(f"  라벨(y) 형태: {labels_arr.shape}, dtype: {labels_arr.dtype}")
        print(f"  날짜(meta) 형태: {dates_arr.shape}, dtype: {dates_arr.dtype}")
        print(f"  실제 수익률(actual_return) 형태: {returns_arr.shape}, dtype: {returns_arr.dtype}")
        print(f"  티커(tickers) 형태: {tickers_arr.shape}, dtype: {tickers_arr.dtype}")
        
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
            returns=returns_arr,
            tickers=tickers_arr
        )
        print("저장 완료.")
    else:
        print("생성된 데이터가 없습니다. TXT 파일과 경로를 확인하세요.")


# --- 테스트 함수: 1개 종목만 처리 ---
def test_single_stock(stocks_folder, output_file, n_days, ticker_name=None):
    """
    테스트용: 1개 종목만 처리하여 결과를 확인합니다.
    
    Args:
        stocks_folder: 주식 파일이 있는 폴더 경로
        output_file: 출력 파일명
        n_days: 이미지 윈도우 크기
        ticker_name: 처리할 티커명 (예: 'aap.us'). None이면 첫 번째 파일 사용
    """
    if n_days not in IMAGE_DIMS:
        raise ValueError(f"n_days는 {list(IMAGE_DIMS.keys())} 중 하나여야 합니다.")
        
    img_config = IMAGE_DIMS[n_days]
    min_length = n_days + n_days
    
    # 파일 찾기
    search_path = os.path.join(stocks_folder, "**", "*.txt")
    txt_files = glob.glob(search_path, recursive=True)
    
    if not txt_files:
        print(f"경고: '{search_path}' 경로에서 TXT 파일을 찾을 수 없습니다.")
        return
    
    # 티커명이 지정되면 해당 파일 찾기, 아니면 첫 번째 파일 사용
    if ticker_name:
        target_file = None
        for f in txt_files:
            if os.path.splitext(os.path.basename(f))[0] == ticker_name:
                target_file = f
                break
        if target_file is None:
            print(f"경고: '{ticker_name}' 티커를 찾을 수 없습니다.")
            return
    else:
        target_file = txt_files[0]
        ticker_name = os.path.splitext(os.path.basename(target_file))[0]
    
    print(f"테스트: '{ticker_name}' 종목 처리 중...")
    print(f"파일 경로: {target_file}")
    
    # 단일 파일 처리
    results = process_single_file(target_file, n_days, img_config, min_length)
    
    if not results:
        print("경고: 처리된 데이터가 없습니다.")
        return
    
    print(f"총 {len(results)}개의 (이미지, 라벨) 쌍이 생성되었습니다.")
    
    # 결과를 배열로 변환
    all_images = []
    all_labels = []
    all_dates = []
    all_actual_returns = []
    all_tickers = []
    
    for image, label, date, actual_return, ticker in results:
        all_images.append(image)
        all_labels.append(label)
        all_dates.append(date.strftime('%Y-%m-%d'))
        all_actual_returns.append(actual_return)
        all_tickers.append(ticker)
    
    # NumPy 배열로 변환
    images_arr = np.array(all_images, dtype=np.uint8)
    labels_arr = np.array(all_labels, dtype=np.uint8)
    dates_arr = np.array(all_dates)
    returns_arr = np.array(all_actual_returns, dtype=np.float32)
    tickers_arr = np.array(all_tickers)
    
    # 채널 차원 추가
    images_arr = np.expand_dims(images_arr, axis=-1)
    
    print(f"  이미지(X) 형태: {images_arr.shape}, dtype: {images_arr.dtype}")
    print(f"  라벨(y) 형태: {labels_arr.shape}, dtype: {labels_arr.dtype}")
    print(f"  날짜(meta) 형태: {dates_arr.shape}, dtype: {dates_arr.dtype}")
    print(f"  실제 수익률(actual_return) 형태: {returns_arr.shape}, dtype: {returns_arr.dtype}")
    print(f"  티커(tickers) 형태: {tickers_arr.shape}, dtype: {tickers_arr.dtype}")
    
    # 샘플 정보 출력
    print(f"\n--- 샘플 정보 (처음 5개) ---")
    for i in range(min(5, len(results))):
        print(f"  [{i}] 날짜: {all_dates[i]}, 라벨: {all_labels[i]}, 수익률: {all_actual_returns[i]:.4f}")
    
    # 파일 저장
    print(f"\n데이터를 '{output_file}' 파일로 저장 중...")
    np.savez_compressed(
        output_file,
        images=images_arr,
        labels=labels_arr,
        dates=dates_arr,
        returns=returns_arr,
        tickers=tickers_arr
    )
    print("저장 완료.")


# --- 4. 메인 코드 실행 ---
if __name__ == "__main__":
    # Colab에서 multiprocessing을 사용하려면 'fork' 대신 'spawn'을 사용해야 할 수 있습니다.
    # (일반적으로 Python 3.8+ Linux에서는 'fork'가 기본값)
    if platform.system() != 'Windows':
         from multiprocessing import set_start_method
         try:
             set_start_method('spawn')
         except RuntimeError:
             pass # 이미 설정되었을 수 있음

    # === 테스트: 1개 종목만 처리 ===
    # print("\n--- 테스트: 1개 종목 처리 ---")
    # test_single_stock(
    #     stocks_folder='nyse_nasdaq_nyse_20171011/Stocks',
    #     output_file='data_L5_R5_nyse_test.npz',  # 테스트용 파일명
    #     n_days=5,
    #     ticker_name='aap.us'  # None이면 첫 번째 파일 사용
    # )
    
    # === 전체 파일 처리 (주석 해제하여 사용) ===
    print("\n--- 4. 메인 파이프라인 실행 (5일 예제) ---")
    process_all_files(
        stocks_folder='nyse_nasdaq_nyse_20171011/Stocks', # NYSE/NASDAQ TXT 파일들
        output_file='data_L5_R5_nyse.npz', # 새 이름으로 저장
        n_days=5
    )

