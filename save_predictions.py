import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Input, Conv2D, LeakyReLU, BatchNormalization, 
    MaxPooling2D, Flatten, Dropout, Dense
)
import numpy as np
import pandas as pd
import os

# --- 1. 상수 정의 ---
NPZ_FILE = 'data_L5_R5.npz'  # 이전 단계에서 생성한 NPZ 파일
IMAGE_SHAPE = (32, 15, 1)     # 5-day 이미지 크기 (Height, Width, Channels)
NUM_CLASSES = 2               # 'Up' / 'Down' (2개 클래스)
MODEL_SAVE_PATH = 'cnn_L5_R5_model_final.keras' # 훈련된 모델 저장 경로
STOCKS_COMBINED_CSV = 'nasdaq_yfinance_20200401/stocks_combined.csv'  # 주식 데이터 CSV 파일
NASDAQ_SYMBOLS_FILE = 'nasdaq_yfinance_20200401/nasdaq_stocks_symbols.txt'  # NASDAQ 티커 목록 파일
STALE_STOCKS_FILE = 'stale_stocks_30days_plus.csv'  # Stale stocks 필터링 파일

# --- 2. CNN 모델 구축 함수 ---

def build_model(input_shape, num_classes):
    """
    논문의 Figure 3 (5-day image) 아키텍처에 기반한 CNN 모델을 구축합니다.
    """
    print("CNN 모델 구축 시작...")
    
    model = Sequential(name="Stock_CNN_L5")

    # 입력 레이어
    model.add(Input(shape=input_shape))

    # --- Block 1 (논문 Figure 3 참조) ---
    # 5x3 conv, 64
    model.add(Conv2D(64, (5, 3), padding='same', name='conv1'))
    # Batch Normalization (논문 442번)
    model.add(BatchNormalization(name='bn1'))
    # LReLU (논문 1395번, k=0.01)
    model.add(LeakyReLU(alpha=0.01, name='lrelu1'))
    # 2x1 Max-Pool
    model.add(MaxPooling2D(pool_size=(2, 1), name='pool1'))

    # --- Block 2 (논문 Figure 3 참조) ---
    # 5x3 conv, 128
    model.add(Conv2D(128, (5, 3), padding='same', name='conv2'))
    # Batch Normalization (논문 442번)
    model.add(BatchNormalization(name='bn2'))
    # LReLU (논문 1395번, k=0.01)
    model.add(LeakyReLU(alpha=0.01, name='lrelu2'))
    # 2x1 Max-Pool
    model.add(MaxPooling2D(pool_size=(2, 1), name='pool2'))

    # --- FC Head (Fully Connected Head) ---
    # Flatten (Pool2의 출력: 8 x 15 x 128 = 15360)
    # (논문 Figure 3의 'FC 15360'은 Flatten된 크기를 의미)
    model.add(Flatten(name='flatten'))
    
    # Dropout (논문 443번: 50% Dropout)
    model.add(Dropout(0.5, name='dropout'))

    # Output Layer (논문 275번, 340번)
    # 'Up', 'Down' 2개 클래스에 대한 확률을 Softmax로 출력
    model.add(Dense(num_classes, activation='softmax', name='output'))

    print("모델 구축 완료.")
    return model

# --- 3. 테스트 데이터 로드 함수 ---

def load_test_data(npz_path):
    """
    .npz 파일에서 테스트 데이터(2001-01-01 이후)만 로드합니다.
    """
    print(f"'{npz_path}' 파일에서 테스트 데이터 로드 중...")
    
    if not os.path.exists(npz_path):
        print(f"오류: '{npz_path}' 파일을 찾을 수 없습니다.")
        print("데이터 생성 스크립트를 먼저 실행해 주세요.")
        return None, None, None

    with np.load(npz_path, allow_pickle=True) as data:
        images = data['images']
        dates = data['dates']
        tickers = data['tickers']  # 티커 정보도 로드

    print("데이터 로드 완료.")
    print(f"  이미지 (X) 형태: {images.shape}")
    print(f"  티커 (tickers) 형태: {tickers.shape}")

    # 날짜를 datetime으로 변환
    try:
        dates_np = dates.astype('datetime64[D]')
    except ValueError:
        print("오류: 'dates' 배열을 날짜로 변환할 수 없습니다. .npz 파일을 확인하세요.")
        return None, None, None

    # 논문 기준 분할 날짜 
    # 2001년 1월 1일 이후 데이터만 테스트 데이터로 사용
    split_date = np.datetime64('2001-01-01')
    test_mask = (dates_np >= split_date)

    # 테스트 데이터만 추출
    X_test = images[test_mask]
    dates_test = dates[test_mask]
    tickers_test = tickers[test_mask]

    print(f"테스트 데이터 (2001- 이후): {X_test.shape[0]}개 샘플")

    # 데이터 타입 변경 (메모리 효율성 및 GPU 호환성)
    X_test = X_test.astype('uint8')
    
    return X_test, dates_test, tickers_test

# --- 4. NASDAQ 티커 필터링 함수 ---

def load_nasdaq_symbols(symbols_file):
    """
    NASDAQ 티커 목록 파일에서 티커를 로드합니다.
    """
    print(f"'{symbols_file}' 파일에서 NASDAQ 티커 목록 로드 중...")
    
    if not os.path.exists(symbols_file):
        print(f"경고: '{symbols_file}' 파일을 찾을 수 없습니다. 필터링을 건너뜁니다.")
        return None
    
    with open(symbols_file, 'r') as f:
        symbols = set(line.strip().upper() for line in f if line.strip())
    
    print(f"NASDAQ 티커 {len(symbols)}개 로드 완료.")
    return symbols

def load_stale_stocks_exclusions(stale_stocks_file):
    """
    stale_stocks_30days_plus.csv에서 max_consecutive_days >= 60인 티커 목록을 로드합니다.
    """
    print(f"'{stale_stocks_file}' 파일에서 stale stocks 제외 목록 로드 중...")
    
    if not os.path.exists(stale_stocks_file):
        print(f"경고: '{stale_stocks_file}' 파일을 찾을 수 없습니다. 필터링을 건너뜁니다.")
        return None
    
    try:
        stale_df = pd.read_csv(stale_stocks_file)
        # max_consecutive_days >= 60인 티커만 추출
        excluded_tickers = stale_df[stale_df['max_consecutive_days'] >= 60]['ticker'].unique()
        excluded_tickers_set = set(ticker.upper() for ticker in excluded_tickers)
        
        print(f"제외할 stale stocks 티커 {len(excluded_tickers_set)}개 로드 완료.")
        return excluded_tickers_set
    except Exception as e:
        print(f"경고: '{stale_stocks_file}' 파일을 읽는 중 오류가 발생했습니다: {e}")
        print("필터링을 건너뜁니다.")
        return None

def filter_by_nasdaq_symbols(X_test, dates_test, tickers_test, nasdaq_symbols):
    """
    NASDAQ 티커 목록에 해당하는 데이터만 필터링합니다.
    """
    if nasdaq_symbols is None:
        print("NASDAQ 티커 목록이 없어 필터링을 건너뜁니다.")
        return X_test, dates_test, tickers_test
    
    print("NASDAQ 티커 목록에 따라 데이터 필터링 중...")
    
    # 티커를 대문자로 변환하여 비교
    tickers_test_upper = np.array([ticker.upper() if isinstance(ticker, str) else str(ticker).upper() for ticker in tickers_test])
    
    # NASDAQ 티커 목록에 있는 티커만 필터링
    filter_mask = np.array([ticker in nasdaq_symbols for ticker in tickers_test_upper])
    
    X_test_filtered = X_test[filter_mask]
    dates_test_filtered = dates_test[filter_mask]
    tickers_test_filtered = tickers_test[filter_mask]
    
    print(f"필터링 전: {len(X_test)}개 샘플")
    print(f"필터링 후: {len(X_test_filtered)}개 샘플 (NASDAQ 티커만)")
    
    return X_test_filtered, dates_test_filtered, tickers_test_filtered

def filter_by_stale_stocks(X_test, dates_test, tickers_test, excluded_tickers):
    """
    max_consecutive_days >= 60인 stale stocks를 제외합니다.
    """
    if excluded_tickers is None:
        print("Stale stocks 제외 목록이 없어 필터링을 건너뜁니다.")
        return X_test, dates_test, tickers_test
    
    print("Stale stocks (max_consecutive_days >= 60) 제외 필터링 중...")
    
    # 티커를 대문자로 변환하여 비교
    tickers_test_upper = np.array([ticker.upper() if isinstance(ticker, str) else str(ticker).upper() for ticker in tickers_test])
    
    # 제외 목록에 없는 티커만 필터링
    filter_mask = np.array([ticker not in excluded_tickers for ticker in tickers_test_upper])
    
    X_test_filtered = X_test[filter_mask]
    dates_test_filtered = dates_test[filter_mask]
    tickers_test_filtered = tickers_test[filter_mask]
    
    print(f"필터링 전: {len(X_test)}개 샘플")
    print(f"필터링 후: {len(X_test_filtered)}개 샘플 (stale stocks 제외)")
    
    return X_test_filtered, dates_test_filtered, tickers_test_filtered

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. 모델 구축 및 로드
    print("--- 모델 로드 중... ---")
    model = build_model(IMAGE_SHAPE, NUM_CLASSES)
    
    # 모델 컴파일 (가중치 로드를 위해 필요)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # 저장된 모델 가중치 로드
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"오류: '{MODEL_SAVE_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 train.py를 실행하여 모델을 훈련하고 저장해 주세요.")
        exit()
    
    print(f"'{MODEL_SAVE_PATH}' 파일에서 모델 가중치 로드 중...")
    model.load_weights(MODEL_SAVE_PATH)
    print("모델 로드 완료!")
    
    # 2. 테스트 데이터 로드
    X_test, dates_test, tickers_test = load_test_data(NPZ_FILE)
    
    if X_test is None:
        exit()  # 데이터 로드 실패 시 종료
    
    # 2-1. NASDAQ 티커 목록 로드 및 필터링
    nasdaq_symbols = load_nasdaq_symbols(NASDAQ_SYMBOLS_FILE)
    X_test, dates_test, tickers_test = filter_by_nasdaq_symbols(X_test, dates_test, tickers_test, nasdaq_symbols)
    
    # 2-2. Stale stocks (max_consecutive_days >= 60) 제외 필터링
    excluded_stale_tickers = load_stale_stocks_exclusions(STALE_STOCKS_FILE)
    X_test, dates_test, tickers_test = filter_by_stale_stocks(X_test, dates_test, tickers_test, excluded_stale_tickers)
    
    # --- 3. 테스트 예측 확률을 stocks_combined.csv에 저장 ---
    print("\n--- 테스트 예측 확률을 stocks_combined.csv에 저장 중... ---")
    
    # 테스트 데이터에 대한 예측 확률 계산 (softmax 출력)
    print("테스트 데이터에 대한 예측 확률 계산 중...")
    predictions = model.predict(X_test, batch_size=128, verbose=1)
    # predictions는 (N, 2) 형태: [prob_down, prob_up]
    # probability_up은 두 번째 컬럼 (인덱스 1)
    probability_up = predictions[:, 1]
    
    print(f"예측 확률 계산 완료. 총 {len(probability_up)}개 샘플")
    
    # 예측 결과를 DataFrame으로 만들기
    # dates_test는 문자열 형태일 수 있으므로 datetime으로 변환
    dates_test_str = dates_test.astype(str)  # 문자열로 변환
    predictions_df = pd.DataFrame({
        'ticker': tickers_test,
        'date': dates_test_str,
        'probability_up': probability_up
    })
    
    # stocks_combined.csv 파일 처리
    if os.path.exists(STOCKS_COMBINED_CSV):
        print(f"'{STOCKS_COMBINED_CSV}' 파일 로드 중...")
        
        # 날짜로 먼저 필터링하기 위해 청크 단위로 읽기
        chunk_size = 100000  # 청크 크기
        test_chunks = []  # 테스트 데이터 (2001-01-01 이후)
        train_chunks = []  # 훈련 데이터 (2001-01-01 이전)
        
        print("CSV 파일을 청크 단위로 읽는 중 (날짜로 필터링)...")
        for chunk in pd.read_csv(STOCKS_COMBINED_CSV, chunksize=chunk_size):
            # 날짜를 datetime으로 변환
            chunk['date'] = pd.to_datetime(chunk['date'])
            
            # probability_up 컬럼이 없으면 추가 (빈 값으로)
            if 'probability_up' not in chunk.columns:
                chunk['probability_up'] = np.nan
            
            # 날짜로 분리
            test_chunk = chunk[chunk['date'] >= '2001-01-01'].copy()
            train_chunk = chunk[chunk['date'] < '2001-01-01'].copy()
            
            if len(test_chunk) > 0:
                test_chunks.append(test_chunk)
            if len(train_chunk) > 0:
                train_chunks.append(train_chunk)
        
        # 테스트 데이터 처리
        if test_chunks:
            # 청크들을 합치기
            stocks_test_df = pd.concat(test_chunks, ignore_index=True)
            print(f"테스트 데이터 (2001- 이후): {len(stocks_test_df)}개 행")
            
            # 예측 결과를 stocks_test_df에 병합
            # ticker와 date를 기준으로 매칭
            print("예측 결과를 stocks_combined.csv에 병합 중...")
            
            # date를 문자열로 변환하여 매칭 (형식 통일)
            stocks_test_df['date_str'] = stocks_test_df['date'].dt.strftime('%Y-%m-%d')
            predictions_df['date_str'] = predictions_df['date']
            
            # ticker와 date_str을 기준으로 병합
            merged_test = stocks_test_df.merge(
                predictions_df[['ticker', 'date_str', 'probability_up']],
                on=['ticker', 'date_str'],
                how='left',
                suffixes=('', '_new')
            )
            
            # probability_up_new가 있으면 그것으로 업데이트, 없으면 기존 값 유지
            merged_test['probability_up'] = merged_test['probability_up_new'].fillna(merged_test['probability_up'])
            
            # 불필요한 컬럼 제거
            merged_test = merged_test.drop(columns=['date_str', 'probability_up_new'])
        else:
            merged_test = pd.DataFrame()
            print("경고: 2001-01-01 이후 데이터가 없습니다.")
        
        # 훈련 데이터 처리
        if train_chunks:
            stocks_train_df = pd.concat(train_chunks, ignore_index=True)
            print(f"훈련 데이터 (2001- 이전): {len(stocks_train_df)}개 행")
        else:
            stocks_train_df = pd.DataFrame()
        
        # 훈련 데이터와 테스트 데이터 합치기
        if len(merged_test) > 0 or len(stocks_train_df) > 0:
            if len(merged_test) > 0 and len(stocks_train_df) > 0:
                final_df = pd.concat([stocks_train_df, merged_test], ignore_index=True)
            elif len(merged_test) > 0:
                final_df = merged_test
            else:
                final_df = stocks_train_df
            
            # 날짜 순으로 정렬
            final_df = final_df.sort_values(['ticker', 'date']).reset_index(drop=True)
            
            # CSV 파일로 저장
            print(f"업데이트된 데이터를 '{STOCKS_COMBINED_CSV}' 파일로 저장 중...")
            final_df.to_csv(STOCKS_COMBINED_CSV, index=False)
            print("저장 완료!")
            
            # 통계 출력
            total_rows = len(final_df)
            test_rows_with_prob = final_df[final_df['date'] >= '2001-01-01']['probability_up'].notna().sum() if len(final_df) > 0 else 0
            print(f"\n통계:")
            print(f"  전체 행 수: {total_rows:,}")
            print(f"  테스트 데이터(2001- 이후)에 probability_up이 있는 행 수: {test_rows_with_prob:,}")
        else:
            print("경고: CSV 파일에 데이터가 없습니다.")
    else:
        print(f"경고: '{STOCKS_COMBINED_CSV}' 파일을 찾을 수 없습니다.")

