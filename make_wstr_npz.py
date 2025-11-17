import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# --- 1. 헬퍼 함수 ---

def preprocess_data(df):
    if 'Date' not in df.columns:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df[df.index >= '1993-01-01']
    if len(df) < 10:
        return None
    
    # ⭐️ 핵심: 'AdjReturn' (일간 수익률)만 계산
    df['AdjReturn'] = df['Adj Close'].pct_change()
    df = df.iloc[1:]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[['AdjReturn']]

def process_single_file(filepath, n_days):
    """
    단일 CSV를 처리하여 (날짜, 과거수익률, 미래수익률)을 반환
    """
    results = []
    try:
        df = pd.read_csv(filepath)
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) < (n_days * 2 + 1):
            return []
            
        # ⭐️ 1. WSTR 신호 계산 (과거 n일 누적 수익률)
        # (T-5 부터 T-1 까지의 수익률)
        df_processed['signal_wstr'] = (1 + df_processed['AdjReturn']).rolling(window=n_days).apply(np.prod, raw=True) - 1
        # (shift(-1)로 T-1 기준 신호로 만듦)
        df_processed['signal_wstr'] = df_processed['signal_wstr'].shift(1)

        # ⭐️ 2. 실제 수익률 계산 (미래 n일 누적 수익률)
        # (T+1 부터 T+5 까지의 수익률)
        df_processed['actual_return'] = (1 + df_processed['AdjReturn'].shift(-n_days)).rolling(window=n_days).apply(np.prod, raw=True) - 1
        
        # 날짜(T) 기준으로 모든 데이터 정렬
        df_final = df_processed[['signal_wstr', 'actual_return']].copy()
        df_final = df_final.dropna()
        
        # (날짜, 시그널, 실제수익률) 튜플 리스트로 변환
        return list(zip(df_final.index, df_final['signal_wstr'], df_final['actual_return']))

    except Exception as e:
        return []

# --- 2. 메인 파이프라인 ---

def process_all_files(stocks_folder, output_file, n_days=5, num_workers=None):
    search_path = os.path.join(stocks_folder, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    if not csv_files:
        print(f"경고: '{search_path}' 경로에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일로 벤치마크 데이터 생성 (n_days={n_days})...")
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    all_dates = []
    all_signals = []
    all_returns = []

    process_func = partial(process_single_file, n_days=n_days)
    
    with Pool(processes=num_workers) as pool:
        results_list = list(tqdm(
            pool.imap(process_func, csv_files),
            total=len(csv_files),
            desc="CSV 파일 처리 중"
        ))
    
    # 결과 수집
    for results in results_list:
        for date, signal, actual_return in results:
            # 🚨🚨🚨 여기가 핵심 수정 사항입니다! 🚨🚨🚨
            # Timestamp 객체 대신, 'YYYY-MM-DD' 형식의 문자열로 저장합니다.
            all_dates.append(date.strftime('%Y-%m-%d'))
            all_signals.append(signal)
            all_returns.append(actual_return)

    print("\n모든 파일 처리 완료. NumPy 배열로 변환 중...")
    
    dates_arr = np.array(all_dates)
    signals_arr = np.array(all_signals, dtype='float32')
    returns_arr = np.array(all_returns, dtype='float32')
    
    print(f"총 {len(dates_arr)}개의 (날짜, 시그널, 수익률) 쌍이 생성되었습니다.")
    if len(dates_arr) > 0:
        print(f"  날짜(dates) 형태: {dates_arr.shape}")
        print(f"  시그널(signals) 형태: {signals_arr.shape}")
        print(f"  수익률(returns) 형태: {returns_arr.shape}")

        print(f"데이터를 '{output_file}' 파일로 저장 중...")
        np.savez_compressed(
            output_file,
            dates=dates_arr,
            signals=signals_arr, # WSTR 시그널 저장
            returns=returns_arr
        )
        print("저장 완료.")
    else:
        print("생성된 데이터가 없습니다.")

# --- 3. 메인 코드 실행 ---
if __name__ == "__main__":
    print("--- 5일 WSTR 벤치마크 데이터 생성 시작 ---")
    process_all_files(
        stocks_folder='nasdaq_yfinance_20200401/stocks', # 실제 경로로 변경
        output_file='benchmark_data_WSTR_FIXED.npz', # WSTR 벤치마크 파일
        n_days=5
    )