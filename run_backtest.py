import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import os

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ⭐️ 불러올 예측 답안지 파일
PREDICTIONS_FILE = 'backtest_data_with_predictions.parquet'

# --- 메인 실행 ---
if __name__ == "__main__":

    # ⭐️⭐️⭐️ [핵심] 예측 결과를 파일에서 바로 로드 ⭐️⭐️⭐️
    print(f"\n--- 저장된 예측 결과 ({PREDICTIONS_FILE}) 로드 중 ---")
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"오류: '{PREDICTIONS_FILE}'이 없습니다.")
        print("먼저 run_predictions.py를 실행하여 예측 파일을 생성해야 합니다.")
        exit()
        
    df = pd.read_parquet(PREDICTIONS_FILE)
    print("예측 데이터 로드 완료.")
    print(df.head())

    # 6. 롱숏(Long-Short) 포트폴리오 백테스트 실행 (원본과 동일)
    print("\n--- 롱숏 포트폴리오 백테스트 (주간 리밸런싱 + 극단값 제거) ---")
    
    daily_groups = df.groupby(df.index)
    strategy_returns = [] 
    
    for date, group in tqdm(daily_groups, desc="백테스트 진행 중"):
        
        # 1. 주간 리밸런싱 (월요일에만 실행)
        if date.weekday() != 0:
            continue
            
        if len(group) < 10:
            continue
            
        # Winsorization (극단값 제거)
        q_01 = group['actual_return'].quantile(0.01)
        q_99 = group['actual_return'].quantile(0.99)
        
        group['actual_return_clipped'] = group['actual_return'].clip(lower=q_01, upper=q_99)
        
        # 2. 10분위 계산 (시그널 기준)
        try:
            group['decile'] = pd.qcut(group['signal_prob'], 10, labels=False, duplicates='drop')
        except ValueError:
            continue 

        # 3. 포트폴리오 수익률 계산 (클린 데이터 기준)
        long_return = group[group['decile'] == 9]['actual_return_clipped'].mean()
        short_return = group[group['decile'] == 0]['actual_return_clipped'].mean()
        
        if pd.isna(long_return) or pd.isna(short_return):
            continue

        weekly_strategy_return = long_return - short_return
        strategy_returns.append(pd.Series([weekly_strategy_return], index=[date]))

    # 7. 최종 성과 분석 (원본과 동일)
    print("\n--- 백테스트 결과 ---")
    
    if not strategy_returns:
        print("오류: 백테스트 수익률이 계산되지 않았습니다.")
        exit()

    weekly_returns = pd.concat(strategy_returns)
    clipped_weekly_returns = weekly_returns.clip(lower=-0.99)
    cumulative_returns = (1 + clipped_weekly_returns).cumprod()

    mean_weekly_return = weekly_returns.mean()
    std_weekly_return = weekly_returns.std()
    
    annualized_sharpe_ratio = 0.0
    if std_weekly_return > 0:
        annualized_sharpe_ratio = (mean_weekly_return / std_weekly_return) * np.sqrt(52)

    print(f"테스트 기간: 2001-01-01 ~ 2019-12-31 (데이터 기준)")
    print(f"총 주간 수익률 평균: {mean_weekly_return*100:.4f} %")
    print(f"총 주간 수익률 변동성: {std_weekly_return*100:.4f} %")
    print(f"연간 샤프 비율 (Annualized Sharpe Ratio): {annualized_sharpe_ratio:.4f}")
    
    # 8. 누적 수익률 그래프 시각화 (원본과 동일)
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('CNN 롱숏 포트폴리오 누적 수익률 (주간, 극단값 제거, 2001~)')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익 (1$ 기준)')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('cumulative_returns_L5_R5.png')
    print("누적 수익률 그래프가 'cumulative_returns_L5_R5.png'로 저장되었습니다.")