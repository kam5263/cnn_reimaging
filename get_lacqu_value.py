import pandas as pd

# CSV 파일 경로
csv_path = 'nasdaq_yfinance_20200401/stocks_combined.csv'

# CSV 파일 읽기 (필요한 컬럼만 읽어서 메모리 효율성 향상)
df = pd.read_csv(csv_path)
df_test = df[df['date'] >= '2001-01-01']
print(df_test.head())
print(df_test.tail())
# LACQU 종목의 2020-03-18 데이터 필터링
# result = df[(df['ticker'] == 'LACQU') & (df['date'] == '2020-03-18')]

# if not result.empty:
#     print("LACQU 종목 2020-03-18 데이터:")
#     print("=" * 50)
#     for col in result.columns:
#         print(f"{col}: {result[col].values[0]}")
    
#     print("\n" + "=" * 50)
#     print("ret1, ret5가 0인 이유 분석:")
#     print("=" * 50)
    
#     # ret1, ret5는 미래 수익률입니다
#     # ret1 = (1일 후 가격 / 현재 가격) - 1
#     # ret5 = (5일 후 가격 / 현재 가격) - 1
    
#     target_date = '2020-03-18'
#     target_data = result.iloc[0]
#     current_price = target_data['Adj Close']
    
#     # 1일 후 데이터 확인
#     date_1d_after = df[(df['ticker'] == 'LACQU') & (df['date'] == '2020-03-19')]
#     if not date_1d_after.empty:
#         price_1d_after = date_1d_after.iloc[0]['Adj Close']
#         ret1_calc = (price_1d_after / current_price) - 1.0
#         print(f"\n1일 후 (2020-03-19) 가격: {price_1d_after}")
#         print(f"현재 (2020-03-18) 가격: {current_price}")
#         print(f"ret1 계산: ({price_1d_after} / {current_price}) - 1 = {ret1_calc}")
    
#     # 5일 후 데이터 확인 (주말 제외하면 약 5영업일 후)
#     date_5d_after = df[(df['ticker'] == 'LACQU') & (df['date'] == '2020-03-23')]
#     if not date_5d_after.empty:
#         price_5d_after = date_5d_after.iloc[0]['Adj Close']
#         ret5_calc = (price_5d_after / current_price) - 1.0
#         print(f"\n5일 후 (2020-03-23) 가격: {price_5d_after}")
#         print(f"현재 (2020-03-18) 가격: {current_price}")
#         print(f"ret5 계산: ({price_5d_after} / {current_price}) - 1 = {ret5_calc}")
    
#     print("\n결론: 가격 변동이 없어서 수익률이 0입니다.")
# else:
#     print("해당 날짜의 LACQU 데이터를 찾을 수 없습니다.")

