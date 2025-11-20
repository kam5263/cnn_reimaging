import csv
import os

# 입력 파일 경로
input_file = 'nasdaq_yfinance_20200401/stocks_combined.csv'
# 출력 파일 경로
output_file = 'aapl_prob_up_2001_after.csv'

# 헤더 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    # 출력 파일에 헤더 쓰기
    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        
        # 데이터 필터링
        count = 0
        for row in reader:
            # ticker가 AAPL이고, date가 2001-01-01 이후이고, probability_up이 있는 경우
            if len(row) >= 13:  # 최소 컬럼 수 확인
                ticker = row[0].strip()
                date = row[1].strip()
                prob_up = row[12].strip() if len(row) > 12 else ''
                
                # 조건 확인: AAPL, 2001-01-01 이후, probability_up 값이 있음
                if (ticker.upper() == 'AAPL' and 
                    date >= '2001-01-01' and 
                    prob_up != '' and 
                    prob_up.lower() != 'null'):
                    writer.writerow(row)
                    count += 1
                    
                    # 진행 상황 출력 (1000개마다)
                    if count % 1000 == 0:
                        print(f'Processed {count} rows...')

print(f'완료! 총 {count}개의 행이 {output_file}에 저장되었습니다.')

