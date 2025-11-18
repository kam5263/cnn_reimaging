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
    yfinance CSV를 논문에 맞게 전처리합니다.
    1. 날짜 인덱스 설정
    2. 조정 수익률(AdjReturn) 계산
    3. O/H/L 가격을 종가 대비 비율(factor)로 계산
    """
    # yfinance 데이터 형식에 맞춤
    if 'Date' not in df.columns:
        # print("경고: 'Date' 컬럼이 CSV 파일에 없습니다.")
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

    # 6. 이평선 그리기 (점으로 하나씩)
    # for t in range(n_days):
    #     if not np.isnan(y_ma_arr[t]):
    #         image[y_ma_arr[t], x_center_arr[t]] = 128  # 회색 점
    
    # 이평선 그리기 (흰색으로)
    for t in range(n_days):
        if not np.isnan(y_ma_arr[t]):
            image[y_ma_arr[t], x_center_arr[t]] = 255  # 흰색 점

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
    단일 CSV 파일을 처리하여 (이미지, 라벨, 날짜, 수익률, 티커) 리스트를 반환합니다.
    멀티프로세싱을 위해 분리된 함수입니다.
    """
    results = []
    
    # 파일 경로에서 티커 추출 (예: 'nasdaq_yfinance_20200401/stocks/AAPL.csv' -> 'AAPL')
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
    지정된 폴더의 모든 CSV를 읽어 (이미지, 라벨, 날짜) 쌍을 생성하고
    하나의 .npz 파일로 저장합니다.
    멀티프로세싱을 사용하여 속도를 향상시킵니다.
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
    all_tickers = []
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
        print("생성된 데이터가 없습니다. CSV 파일과 경로를 확인하세요.")


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

    # 5일 이미지/라벨
    print("\n--- 4. 메인 파이프라인 실행 (5일 예제) ---")
    process_all_files(
        stocks_folder='nasdaq_yfinance_20200401/stocks', # 5000개 이상의 csv 파일 존재
        output_file='data_L5_R5.npz', # 새 이름으로 저장
        n_days=5
    )
    # 샘플 이미지 저장
    # import matplotlib.pyplot as plt

    # # 1. 불러올 NPZ 파일 이름 (수정된 파일)
    # # (이 파일이 create_dataset_fixed.py와 동일한 경로에 있다고 가정)
    # NPZ_FILE = 'data_L5_R5_appl.npz' 

    # # 2. 확인할 랜덤 샘플 개수
    # NUM_SAMPLES = 5

    # data = None # data 객체를 try/finally에서 모두 접근할 수 있도록 초기화

    # try:
    #     print(f"'{NPZ_FILE}' 파일 로드 중 (mmap_mode='r')...")
    #     # mmap_mode='r' : 파일을 메모리에 올리지 않고, 디스크에 연결만 합니다.
    #     data = np.load(NPZ_FILE, allow_pickle=True, mmap_mode='r')
        
    #     # 3. 데이터 배열 '포인터' 가져오기 (이 시점엔 메모리 차지 안 함)
    #     images = data['images']
    #     labels = data['labels']
    #     dates = data['dates']
    #     returns = data['returns']
        
    #     total_count = len(images)
        
    #     if total_count == 0:
    #         print("오류: 파일에 데이터가 없습니다.")
    #     else:
    #         print(f"파일 로드 성공. 총 {total_count}개의 샘플 발견.")

    #         # 4. 전체 샘플 중 NUM_SAMPLES 개수만큼 랜덤 인덱스 추출
    #         # replace=False : 중복 없이 뽑기
    #         random_indices = np.random.choice(total_count, NUM_SAMPLES, replace=False)
    #         random_indices.sort() # 보기 좋게 정렬
            
    #         print(f"\n--- {NUM_SAMPLES}개의 랜덤 샘플 정보 (인덱스: {random_indices}) ---")

    #         # 5. 랜덤 인덱스를 하나씩 돌면서 "실제로" 데이터 읽기
    #         for i, index in enumerate(random_indices):
    #             print(f"\n--- {i+1}번째 샘플 (전체 인덱스: {index}) ---")
                
    #             # 🚨 이 시점에 디스크에서 딱 해당 인덱스의 데이터만 읽어옵니다.
    #             sample_image = images[index]
    #             sample_label = labels[index]
    #             sample_date = dates[index]
    #             sample_return = returns[index]
                
    #             print(f"  - 날짜 (Date): {sample_date}")
    #             print(f"  - 라벨 (Label): {sample_label} (0=Down, 1=Up)")
    #             print(f"  - 실제 수익률 (Return): {sample_return:.4f}")
    #             print(f"  - 이미지 형태: {sample_image.shape}")
                
    #             # 6. 이미지 시각화
    #             plt.figure(figsize=(6, 4))
    #             # (32, 15, 1) 형태를 (32, 15)로 변경하여 흑백 이미지로 표시
    #             plt.imshow(np.squeeze(sample_image), cmap='gray', aspect='auto')
    #             plt.title(f"Sample Index: {index} | Date: {sample_date} | Label: {sample_label}")
    #             plt.xlabel("Features (Time steps)")
    #             plt.ylabel("Channels (LOB data)")
    #             plt.savefig(f"sample_{index}.png")

    # except FileNotFoundError:
    #     print(f"오류: '{NPZ_FILE}' 파일을 찾을 수 없습니다.")
    # except KeyError:
    #     print("오류: .npz 파일에 'images', 'labels', 'dates', 'returns' 키 중 하나가 없습니다.")
    # except Exception as e:
    #     print(f"파일 로드 중 알 수 없는 오류 발생: {e}")

    # finally:
    #     # 7. (매우 중요) mmap_mode로 열었으면 반드시 닫아주어야 합니다.
    #     if data is not None and hasattr(data, 'close'):
    #         data.close()
    #         print("\n파일 핸들(mmap)을 닫았습니다.")