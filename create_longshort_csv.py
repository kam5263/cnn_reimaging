import pandas as pd
import os

# --- NYSE 데이터 설정 ---
STOCKS_COMBINED_CSV = 'nyse_nasdaq_nyse_20171011/stocks_combined.csv'  # 주식 데이터 CSV 파일
OUTPUT_FILE = 'longshort_nyse_2001_after.csv'  # 출력 파일 경로

print(f"'{STOCKS_COMBINED_CSV}' 파일 로드 중...")

# CSV 파일을 청크 단위로 읽어서 2001-01-01 이후 데이터만 필터링
chunk_size = 100000
chunks = []

print("CSV 파일을 읽고 2001-01-01 이후 데이터 필터링 중...")
for chunk in pd.read_csv(STOCKS_COMBINED_CSV, chunksize=chunk_size):
    # 날짜를 datetime으로 변환
    chunk['date'] = pd.to_datetime(chunk['date'])
    
    # 2001-01-01 이후 데이터만 필터링
    filtered_chunk = chunk[chunk['date'] >= '2001-01-01'].copy()
    
    if len(filtered_chunk) > 0:
        chunks.append(filtered_chunk)
        print(f"  처리된 청크: {len(filtered_chunk)}개 행")

# 모든 청크 합치기
if chunks:
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n총 {len(df)}개 행 로드 완료")
else:
    print("경고: 2001-01-01 이후 데이터가 없습니다.")
    exit(1)

# probability_up이 있는 데이터만 필터링
df = df[df['probability_up'].notna()].copy()
print(f"probability_up이 있는 데이터: {len(df)}개 행")

# 날짜별로 그룹화하여 probability_up이 가장 높은 종목과 가장 낮은 종목 찾기
print("\n날짜별로 Long/Short 종목 선택 중...")

result_rows = []

# 날짜별로 그룹화
for date, group in df.groupby('date'):
    # probability_up이 있는 종목만 필터링
    valid_group = group[group['probability_up'].notna()].copy()
    
    if len(valid_group) == 0:
        continue
    
    # 리밸런싱 당일에 Volume=0인 종목 제외
    if 'Volume' in valid_group.columns:
        valid_group = valid_group[valid_group['Volume'] > 0].copy()
    
    if len(valid_group) == 0:
        continue
    
    # probability_up 기준으로 정렬
    sorted_group = valid_group.sort_values('probability_up', ascending=False)
    
    # Long: probability_up이 가장 높은 종목 (1등)
    long_row = sorted_group.iloc[0]
    
    # Short: probability_up이 가장 낮은 종목 (꼴등)
    short_row = sorted_group.iloc[-1]
    
    # 결과 행 생성
    result_row = {
        'Date': date.strftime('%Y-%m-%d'),
        'Long': long_row['ticker'],
        'long_close': long_row['Close'],
        'long_ret5': long_row['ret5'],
        'long_prob_up': long_row['probability_up'],
        'Short': short_row['ticker'],
        'short_close': short_row['Close'],
        'short_ret5': short_row['ret5'],
        'short_prob_up': short_row['probability_up']
    }
    
    result_rows.append(result_row)

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame(result_rows)

# 날짜 순으로 정렬
result_df = result_df.sort_values('Date').reset_index(drop=True)

# CSV 파일로 저장
print(f"\n결과를 '{OUTPUT_FILE}' 파일로 저장 중...")
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"완료! 총 {len(result_df)}개 날짜의 데이터가 저장되었습니다.")
print(f"\n샘플 데이터 (처음 5개 행):")
print(result_df.head().to_string())

