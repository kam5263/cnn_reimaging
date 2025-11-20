"""
미국 주요 거래소 상장 주식 필터링 스크립트

이 스크립트는 symbols_valid_meta.csv 파일을 분석하여 
Listing Exchange가 Q, N, A, P인 주식만 추출합니다 (ETF 제외).
Q = NASDAQ, N = NYSE, A = NYSE American, P = NYSE Arca

컬럼 설명:
- Nasdaq Traded: Y/N - 나스닥에서 거래되는지 여부
- Symbol: 주식 심볼
- Security Name: 증권명
- Listing Exchange: 상장 거래소
  * Q = NASDAQ
  * N = NYSE (뉴욕증권거래소)
  * A = NYSE American (구 AMEX)
  * P = NYSE Arca
  * Z = BATS
- Market Category: 시장 카테고리 (NASDAQ 상장 주식의 경우)
  * Q = NASDAQ Global Select Market (대형주, 엄격한 상장 기준)
  * G = NASDAQ Global Market (중형주)
  * S = NASDAQ Capital Market (소형주)
  * (빈 값) = NASDAQ에 상장되지 않은 주식
- ETF: Y/N - ETF 여부
- Round Lot Size: 보통 100.0 (원화 단위 거래 단위)
- Test Issue: Y/N - 테스트 이슈 여부 (제외해야 함)
- Financial Status: 재무 상태
  * N = Normal (정상)
  * D = Deficient (부족)
  * E = Delinquent (연체)
  * H = Halted (거래정지)
  * S = Suspended (상장정지)
  * (빈 값) = 상태 불명
- CQS Symbol: Consolidated Quote System 심볼
- NASDAQ Symbol: NASDAQ 심볼
- NextShares: Y/N - NextShares (ETF 유형 중 하나)
"""

import pandas as pd
import os


def analyze_columns(df):
    """컬럼별 통계 정보 출력"""
    print("=" * 80)
    print("컬럼별 통계 분석")
    print("=" * 80)
    
    print(f"\n전체 행 수: {len(df):,}")
    
    print("\n[Listing Exchange - 상장 거래소]")
    print(df['Listing Exchange'].value_counts())
    
    print("\n[Market Category - 시장 카테고리]")
    print(df['Market Category'].value_counts())
    
    print("\n[ETF 여부]")
    print(df['ETF'].value_counts())
    
    print("\n[Test Issue - 테스트 이슈]")
    print(df['Test Issue'].value_counts())
    
    print("\n[Financial Status - 재무 상태]")
    print(df['Financial Status'].value_counts())
    
    print("\n[NextShares]")
    print(df['NextShares'].value_counts())


def filter_nasdaq_stocks(df, 
                         exclude_etf=True,
                         exclude_test_issues=True,
                         exclude_problematic_financial_status=True,
                         allowed_exchanges=['Q', 'N', 'A', 'P']):
    """
    주요 거래소 상장 주식 필터링
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    exclude_etf : bool
        ETF 제외 여부 (기본값: True)
    exclude_test_issues : bool
        테스트 이슈 제외 여부 (기본값: True)
    exclude_problematic_financial_status : bool
        문제가 있는 재무 상태 제외 여부 (기본값: True)
        (D=Deficient, E=Delinquent, H=Halted, S=Suspended 제외)
    allowed_exchanges : list
        허용할 Listing Exchange 목록 (기본값: ['Q', 'N', 'A', 'P'])
        Q = NASDAQ, N = NYSE, A = NYSE American, P = NYSE Arca
    
    Returns:
    --------
    DataFrame
        필터링된 데이터프레임
    """
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    print("\n" + "=" * 80)
    print("필터링 시작")
    print("=" * 80)
    print(f"원본 행 수: {original_count:,}")
    
    # 1. ETF 제외
    if exclude_etf:
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['ETF'] == 'N']
        print(f"\n1. ETF 제외: {before:,} -> {len(filtered_df):,} (제외: {before - len(filtered_df):,})")
    
    # 2. NextShares 제외 (ETF 유형)
    if exclude_etf:
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['NextShares'] == 'N']
        print(f"2. NextShares 제외: {before:,} -> {len(filtered_df):,} (제외: {before - len(filtered_df):,})")
    
    # 3. 테스트 이슈 제외
    if exclude_test_issues:
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Test Issue'] == 'N']
        print(f"3. 테스트 이슈 제외: {before:,} -> {len(filtered_df):,} (제외: {before - len(filtered_df):,})")
    
    # 4. 허용된 거래소만 포함
    if allowed_exchanges:
        before = len(filtered_df)
        # Listing Exchange가 허용된 목록에 있는 경우만 포함
        exchange_condition = filtered_df['Listing Exchange'].isin(allowed_exchanges)
        filtered_df = filtered_df[exchange_condition]
        exchange_names = {
            'Q': 'NASDAQ',
            'N': 'NYSE',
            'A': 'NYSE American',
            'P': 'NYSE Arca',
            'Z': 'BATS'
        }
        allowed_names = [exchange_names.get(e, e) for e in allowed_exchanges]
        print(f"4. 허용된 거래소만 포함 ({', '.join(allowed_names)}): {before:,} -> {len(filtered_df):,} (제외: {before - len(filtered_df):,})")
    
    # 5. 문제가 있는 재무 상태 제외
    if exclude_problematic_financial_status:
        before = len(filtered_df)
        # Financial Status가 'D', 'E', 'H', 'S'인 경우 제외
        # 'N' (Normal) 또는 빈 값만 포함
        problematic_status = ['D', 'E', 'H', 'S']
        filtered_df = filtered_df[
            ~filtered_df['Financial Status'].isin(problematic_status)
        ]
        print(f"5. 문제가 있는 재무 상태 제외: {before:,} -> {len(filtered_df):,} (제외: {before - len(filtered_df):,})")
    
    print("\n" + "=" * 80)
    print(f"최종 필터링 결과: {original_count:,} -> {len(filtered_df):,} (제외: {original_count - len(filtered_df):,})")
    print("=" * 80)
    
    return filtered_df


def main():
    # 파일 경로
    input_file = 'nasdaq_yfinance_20200401/symbols_valid_meta.csv'
    output_file = 'nasdaq_yfinance_20200401/nasdaq_stocks_filtered.csv'
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"오류: 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # CSV 파일 읽기
    print(f"파일 읽는 중: {input_file}")
    df = pd.read_csv(input_file)
    
    # 컬럼 분석
    analyze_columns(df)
    
    # 필터링 (Listing Exchange가 Q, N, A, P인 것만 포함)
    filtered_df = filter_nasdaq_stocks(
        df,
        exclude_etf=True,
        exclude_test_issues=True,
        exclude_problematic_financial_status=True,
        allowed_exchanges=['Q', 'N', 'A', 'P']
    )
    
    # 결과 저장
    print(f"\n필터링된 데이터 저장 중: {output_file}")
    filtered_df.to_csv(output_file, index=False)
    print(f"저장 완료: {len(filtered_df):,}개 행")
    
    # 필터링된 데이터 요약 정보
    print("\n" + "=" * 80)
    print("필터링된 데이터 요약")
    print("=" * 80)
    print(f"\n총 심볼 수: {len(filtered_df):,}")
    print(f"\nMarket Category 분포:")
    print(filtered_df['Market Category'].value_counts())
    print(f"\nListing Exchange 분포:")
    print(filtered_df['Listing Exchange'].value_counts())
    print(f"\nFinancial Status 분포:")
    print(filtered_df['Financial Status'].value_counts())
    
    # 샘플 출력
    print("\n" + "=" * 80)
    print("샘플 데이터 (처음 10개)")
    print("=" * 80)
    print(filtered_df[['Symbol', 'Security Name', 'Listing Exchange', 
                       'Market Category', 'ETF', 'Financial Status']].head(10).to_string())
    
    # 심볼 리스트만 저장 (선택사항)
    symbols_file = 'nasdaq_yfinance_20200401/nasdaq_stocks_symbols.txt'
    with open(symbols_file, 'w') as f:
        for symbol in sorted(filtered_df['Symbol'].unique()):
            f.write(f"{symbol}\n")
    print(f"\n심볼 리스트 저장: {symbols_file} ({len(filtered_df['Symbol'].unique()):,}개)")


if __name__ == '__main__':
    main()

