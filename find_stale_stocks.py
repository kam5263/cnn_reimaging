import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def check_stale_pattern(df):
    """
    Volume이 0이고 모든 가격(Open, High, Low, Close)이 전날과 동일한 행이 반복되는지 확인
    
    Returns:
        dict: 패턴 정보를 담은 딕셔너리
            - has_pattern: 패턴이 있는지 여부
            - consecutive_days: 연속된 일수
            - max_consecutive: 최대 연속 일수
            - total_stale_days: 전체 stale 일수
            - stale_periods: stale 기간 리스트 [(start_date, end_date, days), ...]
    """
    if len(df) < 2:
        return {
            'has_pattern': False,
            'consecutive_days': 0,
            'max_consecutive': 0,
            'total_stale_days': 0,
            'stale_periods': []
        }
    
    # Date를 인덱스로 설정 (이미 인덱스가 아닌 경우)
    if 'Date' in df.columns:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    
    # 1993-01-01 이후 데이터만 사용
    df = df[df.index >= '1993-01-01']
    
    # 데이터가 너무 적으면 건너뜀
    if len(df) < 2:
        return {
            'has_pattern': False,
            'consecutive_days': 0,
            'max_consecutive': 0,
            'total_stale_days': 0,
            'stale_periods': []
        }
    
    # 필요한 컬럼 확인
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return {
            'has_pattern': False,
            'consecutive_days': 0,
            'max_consecutive': 0,
            'total_stale_days': 0,
            'stale_periods': []
        }
    
    # Volume이 0이고 가격이 전날과 동일한 행 찾기
    stale_mask = []
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        
        # Volume이 0이고 모든 가격이 전날과 동일한지 확인
        is_volume_zero = curr_row['Volume'] == 0
        is_price_same = (
            curr_row['Open'] == prev_row['Open'] and
            curr_row['High'] == prev_row['High'] and
            curr_row['Low'] == prev_row['Low'] and
            curr_row['Close'] == prev_row['Close']
        )
        
        stale_mask.append(is_volume_zero and is_price_same)
    
    # 첫 번째 행은 비교할 이전 행이 없으므로 False
    stale_mask = [False] + stale_mask
    
    if not any(stale_mask):
        return {
            'has_pattern': False,
            'consecutive_days': 0,
            'max_consecutive': 0,
            'total_stale_days': 0,
            'stale_periods': []
        }
    
    # 연속된 stale 기간 찾기
    stale_periods = []
    max_consecutive = 0
    current_consecutive = 0
    period_start = None
    
    for i, is_stale in enumerate(stale_mask):
        if is_stale:
            if period_start is None:
                period_start = df.index[i]
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            if period_start is not None:
                # 기간 종료
                period_end = df.index[i-1]
                days = (period_end - period_start).days + 1
                stale_periods.append((period_start, period_end, days))
                period_start = None
            current_consecutive = 0
    
    # 마지막에 기간이 끝나지 않은 경우
    if period_start is not None:
        period_end = df.index[-1]
        days = (period_end - period_start).days + 1
        stale_periods.append((period_start, period_end, days))
    
    total_stale_days = sum(stale_mask)
    
    return {
        'has_pattern': True,
        'consecutive_days': max_consecutive,
        'max_consecutive': max_consecutive,
        'total_stale_days': total_stale_days,
        'stale_periods': stale_periods
    }

def find_stale_stocks(stocks_dir, symbols_file):
    """
    stocks 디렉토리 내의 모든 CSV 파일을 검사하여 stale 패턴을 찾음
    
    Args:
        stocks_dir: stocks 디렉토리 경로
        symbols_file: nasdaq_stocks_symbols.txt 파일 경로 (None이면 필터링 없이 모든 파일 검사)
    
    Returns:
        list: 패턴이 있는 종목들의 정보 리스트
    """
    stocks_path = Path(stocks_dir)
    csv_files = list(stocks_path.glob('*.txt'))
    
    # symbols_file이 있는 경우에만 필터링
    if symbols_file and os.path.exists(symbols_file):
        # nasdaq_stocks_symbols.txt에서 티커 목록 읽기
        with open(symbols_file, 'r') as f:
            valid_tickers = set(line.strip() for line in f if line.strip())
        
        # 유효한 티커에 해당하는 CSV 파일만 필터링
        filtered_files = [f for f in csv_files if f.stem in valid_tickers]
    else:
        # symbols_file이 없으면 모든 CSV 파일 검사
        filtered_files = csv_files
    
    results = []
    
    print(f"총 {len(csv_files)}개의 CSV 파일 중 {len(filtered_files)}개의 파일을 검사합니다...")
    
    for csv_file in tqdm(filtered_files, desc="검사 중"):
        try:
            df = pd.read_csv(csv_file)
            pattern_info = check_stale_pattern(df)
            
            if pattern_info['has_pattern']:
                ticker = csv_file.stem
                results.append({
                    'ticker': ticker,
                    'file': csv_file.name,
                    'max_consecutive_days': pattern_info['max_consecutive'],
                    'total_stale_days': pattern_info['total_stale_days'],
                    'stale_periods': pattern_info['stale_periods'],
                    'num_periods': len(pattern_info['stale_periods'])
                })
        except Exception as e:
            print(f"\n오류 발생 ({csv_file.name}): {e}")
            continue
    
    return results

if __name__ == '__main__':
    #stocks_dir = 'nasdaq_yfinance_20200401/stocks'
    #symbols_file = 'nasdaq_yfinance_20200401/nasdaq_stocks_symbols.txt'
    stocks_dir = 'nyse_nasdaq_nyse_20171011/Stocks'
    symbols_file = None  # None이면 필터링 없이 모든 파일 검사
    print("Stale 패턴을 가진 종목을 찾는 중...")
    print("조건: Volume=0이고 Open/High/Low/Close가 전날과 동일한 행이 반복되는 경우")
    if symbols_file:
        print("필터: nasdaq_stocks_symbols.txt에 있는 티커만 검사, 1993-01-01 이후 데이터만 사용\n")
    else:
        print("필터: 모든 CSV 파일 검사, 1993-01-01 이후 데이터만 사용\n")
    
    results = find_stale_stocks(stocks_dir, symbols_file)
    
    # 결과 정렬 (최대 연속 일수 기준 내림차순)
    results.sort(key=lambda x: x['max_consecutive_days'], reverse=True)
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"총 {len(results)}개의 종목에서 패턴을 발견했습니다.")
    print(f"{'='*80}\n")
    
    # 상위 20개 출력
    print(f"{'Ticker':<10} {'Max Consecutive':<18} {'Total Stale Days':<18} {'Periods':<10}")
    print("-" * 80)
    
    for result in results[:20]:
        print(f"{result['ticker']:<10} {result['max_consecutive_days']:<18} {result['total_stale_days']:<18} {result['num_periods']:<10}")
    
    if len(results) > 20:
        print(f"\n... 외 {len(results) - 20}개 종목")
    
    # 상세 정보를 CSV로 저장
    output_file = 'stale_stocks_detailed.csv'
    detailed_results = []
    for result in results:
        for period in result['stale_periods']:
            detailed_results.append({
                'ticker': result['ticker'],
                'start_date': period[0],
                'end_date': period[1],
                'days': period[2],
                'max_consecutive_days': result['max_consecutive_days'],
                'total_stale_days': result['total_stale_days']
            })
    
    if detailed_results:
        df_output = pd.DataFrame(detailed_results)
        df_output.to_csv(output_file, index=False)
        print(f"\n상세 정보가 '{output_file}'에 저장되었습니다.")
    
    # 요약 정보 저장
    summary_file = 'stale_stocks_summary.csv'
    df_summary = pd.DataFrame(results)
    df_summary['stale_periods'] = df_summary['stale_periods'].apply(lambda x: str(x))
    df_summary.to_csv(summary_file, index=False)
    print(f"요약 정보가 '{summary_file}'에 저장되었습니다.")

