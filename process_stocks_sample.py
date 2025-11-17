import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import platform

def preprocess_data(df):
    """
    yfinance CSV를 전처리합니다.
    1. 날짜 인덱스 설정
    2. 1993-01-01 이후 데이터 필터링
    3. 이동평균 계산 (ma5, ma20)
    4. 미래 수익률(ret1, ret5, ret20) 계산
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

    # 데이터가 너무 적으면 건너뜀 (최소 20일 필요)
    if len(df) < 20:
        return None

    # Close가 0인 경우 처리
    df['Close'] = df['Close'].replace(0, 1e-9)
    df['Adj Close'] = df['Adj Close'].replace(0, 1e-9)
    
    # 이동평균 계산 (5일, 20일 이동평균)
    df['ma5'] = df['Adj Close'].rolling(window=5, min_periods=1).mean()
    df['ma20'] = df['Adj Close'].rolling(window=20, min_periods=1).mean()
    
    # 미래 수익률 계산
    # ret1: 1일 후 수익률
    df['ret1'] = df['Adj Close'].shift(-1) / df['Adj Close'] - 1.0
    
    # ret5: 5일 후 수익률
    df['ret5'] = df['Adj Close'].shift(-5) / df['Adj Close'] - 1.0
    
    # ret20: 20일 후 수익률
    df['ret20'] = df['Adj Close'].shift(-20) / df['Adj Close'] - 1.0
    
    # 무한대/NaN 값 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # ret1, ret5, ret20이 NaN인 행은 제거 (미래 데이터가 없는 마지막 행들)
    # 하지만 다른 컬럼은 유지하기 위해 dropna(subset=...) 사용
    df = df.dropna(subset=['ret1', 'ret5', 'ret20'], how='all')
    
    return df


def process_single_file(filepath, output_folder):
    """
    단일 CSV 파일을 처리하여 전처리된 데이터를 반환합니다.
    """
    try:
        df = pd.read_csv(filepath)
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) == 0:
            return None
        
        # 파일명 추출 (확장자 제외)
        filename = os.path.basename(filepath)
        stock_symbol = os.path.splitext(filename)[0]
        
        # 필요한 컬럼만 선택
        output_df = df_processed[[
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'ma5', 'ma20', 'ret1', 'ret5', 'ret20'
        ]].copy()
        
        # 인덱스(Date)를 컬럼으로 변환
        output_df.reset_index(inplace=True)
        
        # 컬럼명 정리 (Date를 date로)
        output_df.rename(columns={'Date': 'date'}, inplace=True)
        
        # 출력 파일 경로
        output_path = os.path.join(output_folder, filename)
        
        # CSV로 저장
        output_df.to_csv(output_path, index=False)
        
        return stock_symbol, len(output_df)
    
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        print(f"파일 처리 중 오류: {filepath}, {e}")
        return None


def process_all_files(input_folder, output_folder, num_workers=None):
    """
    지정된 폴더의 모든 CSV를 읽어서 전처리하고 저장합니다.
    멀티프로세싱을 사용하여 속도를 향상시킵니다.
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # glob를 사용해 하위 폴더 포함 모든 csv 검색
    search_path = os.path.join(input_folder, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"경고: '{search_path}' 경로에서 CSV 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일 처리 시작...")
    
    # 멀티프로세싱 설정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"멀티프로세싱 사용: {num_workers}개 프로세스")
    
    # 멀티프로세싱으로 파일 처리
    process_func = partial(process_single_file, output_folder=output_folder)
    
    results = []
    with Pool(processes=num_workers) as pool:
        results_list = list(tqdm(
            pool.imap(process_func, csv_files),
            total=len(csv_files),
            desc="CSV 파일 처리 중"
        ))
    
    # 결과 수집
    print("\n멀티프로세싱 완료. 결과 수집 중...")
    successful = 0
    failed = 0
    
    for result in results_list:
        if result is not None:
            stock_symbol, row_count = result
            results.append((stock_symbol, row_count))
            successful += 1
        else:
            failed += 1
    
    print(f"\n처리 완료:")
    print(f"  성공: {successful}개 파일")
    print(f"  실패: {failed}개 파일")
    print(f"  출력 폴더: {output_folder}")
    
    if results:
        total_rows = sum(count for _, count in results)
        print(f"  총 데이터 행 수: {total_rows:,}")
        print(f"\n처리된 주식 샘플 (상위 10개):")
        for symbol, count in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {symbol}: {count:,}행")


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    # Colab에서 multiprocessing을 사용하려면 'fork' 대신 'spawn'을 사용해야 할 수 있습니다.
    if platform.system() != 'Windows':
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass  # 이미 설정되었을 수 있음
    
    print("--- 주식 데이터 전처리 시작 ---")
    process_all_files(
        input_folder='nasdaq_yfinance_20200401/stocks_sample',
        output_folder='nasdaq_yfinance_20200401/stocks_processed',
        num_workers=None  # 자동으로 CPU 코어 수에 맞춤
    )
    print("\n--- 전처리 완료 ---")

