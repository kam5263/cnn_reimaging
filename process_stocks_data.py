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
    NYSE/NASDAQ TXT 파일을 전처리합니다.
    1. 날짜 인덱스 설정
    2. 1993-01-01 이후 데이터 필터링
    3. 이동평균 계산 (ma5, ma20)
    4. 미래 수익률(ret1, ret5, ret20) 계산
    """
    # NYSE/NASDAQ 데이터 형식에 맞춤
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
    
    # 이동평균 계산 (5일, 20일 이동평균)
    # NYSE/NASDAQ 데이터는 이미 adjusted된 Close를 사용
    df['ma5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['ma20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # 미래 수익률 계산
    # ret1: 1일 후 수익률
    df['ret1'] = df['Close'].shift(-1) / df['Close'] - 1.0
    
    # ret5: 5일 후 수익률
    df['ret5'] = df['Close'].shift(-5) / df['Close'] - 1.0
    
    # ret20: 20일 후 수익률
    df['ret20'] = df['Close'].shift(-20) / df['Close'] - 1.0
    
    # 무한대/NaN 값 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # ret1, ret5, ret20이 NaN인 행은 제거 (미래 데이터가 없는 마지막 행들)
    # 하지만 다른 컬럼은 유지하기 위해 dropna(subset=...) 사용
    df = df.dropna(subset=['ret1', 'ret5', 'ret20'], how='all')
    
    return df


def process_single_file(filepath):
    """
    단일 CSV 파일을 처리하여 전처리된 데이터를 반환합니다.
    티커 컬럼을 맨 앞에 추가합니다.
    """
    try:
        df = pd.read_csv(filepath)
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) == 0:
            return None
        
        # 파일명 추출 (확장자 제외) - 티커
        filename = os.path.basename(filepath)
        stock_symbol = os.path.splitext(filename)[0]
        
        # 필요한 컬럼만 선택
        # NYSE/NASDAQ 데이터에는 Adj Close가 없으므로 Close만 사용
        output_df = df_processed[[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'ma5', 'ma20', 'ret1', 'ret5', 'ret20'
        ]].copy()
        
        # 인덱스(Date)를 컬럼으로 변환
        output_df.reset_index(inplace=True)
        
        # 컬럼명 정리 (Date를 date로)
        output_df.rename(columns={'Date': 'date'}, inplace=True)
        
        # 티커 컬럼을 맨 앞에 추가
        output_df.insert(0, 'ticker', stock_symbol)
        
        return output_df
    
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        print(f"파일 처리 중 오류: {filepath}, {e}")
        return None


def process_all_files(input_folder, output_file, num_workers=None):
    """
    지정된 폴더의 모든 CSV를 읽어서 전처리하고 하나의 파일로 합쳐서 저장합니다.
    멀티프로세싱을 사용하여 속도를 향상시킵니다.
    """
    # glob를 사용해 하위 폴더 포함 모든 csv 검색
    search_path = os.path.join(input_folder, "**", "*.txt")
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
    results = []
    with Pool(processes=num_workers) as pool:
        results_list = list(tqdm(
            pool.imap(process_single_file, csv_files),
            total=len(csv_files),
            desc="CSV 파일 처리 중"
        ))
    
    # 결과 수집 및 합치기 (메모리 효율적인 청크 방식)
    print("\n멀티프로세싱 완료. 결과 수집 및 합치기 중...")
    successful = 0
    failed = 0
    ticker_counts = {}
    
    # 첫 번째 유효한 결과로 헤더 작성
    first_df = None
    for result in results_list:
        if result is not None:
            first_df = result
            break
    
    if first_df is None:
        print("처리된 데이터가 없습니다.")
        return
    
    # 청크 단위로 처리하여 메모리 효율성 향상
    CHUNK_SIZE = 100  # 한 번에 합칠 DataFrame 개수
    temp_files = []
    chunk_num = 0
    
    print("청크 단위로 데이터 합치는 중...")
    current_chunk = []
    
    for result in results_list:
        if result is not None:
            current_chunk.append(result)
            ticker = result['ticker'].iloc[0]
            ticker_counts[ticker] = len(result)
            successful += 1
            
            # 청크가 가득 차면 합쳐서 임시 파일로 저장
            if len(current_chunk) >= CHUNK_SIZE:
                chunk_df = pd.concat(current_chunk, ignore_index=True)
                temp_file = f"{output_file}.temp_{chunk_num}.csv"
                chunk_df.to_csv(temp_file, index=False)
                temp_files.append(temp_file)
                current_chunk = []
                chunk_num += 1
                print(f"  청크 {chunk_num} 저장 완료 ({successful}개 파일 처리됨)")
        else:
            failed += 1
    
    # 남은 청크 처리
    if current_chunk:
        chunk_df = pd.concat(current_chunk, ignore_index=True)
        temp_file = f"{output_file}.temp_{chunk_num}.csv"
        chunk_df.to_csv(temp_file, index=False)
        temp_files.append(temp_file)
        chunk_num += 1
        print(f"  마지막 청크 저장 완료")
    
    # 모든 임시 파일을 하나로 합치기
    print(f"모든 청크를 하나의 파일로 합치는 중...")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, temp_file in enumerate(temp_files):
            with open(temp_file, 'r', encoding='utf-8') as infile:
                if i == 0:
                    # 첫 번째 파일은 헤더 포함
                    outfile.write(infile.read())
                    header_written = True
                else:
                    # 나머지 파일은 헤더 제외
                    next(infile)  # 헤더 스킵
                    outfile.write(infile.read())
            # 임시 파일 삭제
            os.remove(temp_file)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(temp_files)} 청크 병합 완료")
    
    # 최종 정렬 (대용량 파일의 경우 메모리 문제가 있을 수 있으므로 선택적)
    print("데이터 정렬 중...")
    try:
        # 파일 크기를 확인하여 정렬 여부 결정
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        if file_size_mb > 500:  # 500MB 이상이면 정렬 스킵 (선택적)
            print(f"  파일 크기가 {file_size_mb:.1f}MB로 큽니다. 정렬을 건너뜁니다.")
            print("  (필요시 나중에 별도로 정렬할 수 있습니다.)")
        else:
            final_df = pd.read_csv(output_file)
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values(['ticker', 'date']).reset_index(drop=True)
            final_df.to_csv(output_file, index=False)
            print("  정렬 완료")
    except MemoryError:
        print("  메모리 부족으로 정렬을 건너뜁니다.")
    
    # 총 행 수 계산
    total_rows = sum(ticker_counts.values())
    
    print(f"\n처리 완료:")
    print(f"  성공: {successful}개 파일")
    print(f"  실패: {failed}개 파일")
    print(f"  출력 파일: {output_file}")
    print(f"  총 데이터 행 수: {total_rows:,}")
    print(f"\n처리된 주식 샘플 (상위 10개):")
    for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {ticker}: {count:,}행")


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
        input_folder='nyse_nasdaq_nyse_20171011/Stocks',
        output_file='nyse_nasdaq_nyse_20171011/stocks_combined.csv',
        num_workers=None  # 자동으로 CPU 코어 수에 맞춤
    )
    print("\n--- 전처리 완료 ---")

