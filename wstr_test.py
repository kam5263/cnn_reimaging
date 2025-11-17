import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# --- 1. 상수 정의 ---
# ⭐️ [수정 1] 로드할 파일 변경
NPZ_FILE = 'benchmark_data_WSTR.npz'

# --- 2. 데이터 로드 함수 (벤치마크용) ---
def load_data_for_backtest(npz_path):
    print(f"'{npz_path}' 파일에서 벤치마크 데이터 로드 중...")
    with np.load(npz_path, allow_pickle=True) as data:
        dates = data['dates']
        signals = data['signals'] # ⭐️ WSTR 시그널
        returns = data['returns'] 
    
    print(f"  시그널 (signals) 형태: {signals.shape}")
    print(f"  수익률 (returns) 형태: {returns.shape}")
    
    return dates, signals, returns

# --- 3. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. 백테스트 데이터 로드
    dates_data, signals_data, returns_data = load_data_for_backtest(NPZ_FILE)

    # 2. 훈련/테스트 데이터 분할
    print("\n--- 테스트 데이터셋 분할 (시계열 기준) ---")
    dates_np = dates_data.astype('datetime64[D]')
    split_date = np.datetime64('2001-01-01')
    test_mask = (dates_np >= split_date)

    dates_test = dates_data[test_mask]
    signals_test = signals_data[test_mask] # ⭐️ WSTR 시그널 테스트셋
    returns_test = returns_data[test_mask] 

    print(f"테스트 (2001-) 데이터: {signals_test.shape[0]}개 샘플")
    del dates_data, signals_data, returns_data, test_mask
    gc.collect()

    # (모델 로드 및 예측 부분은 필요 없으므로 삭제)

    # 4. 백테스트용 데이터프레임 생성
    print("\n--- 백테스트용 데이터프레임 생성 ---")
    df = pd.DataFrame({
        'date': pd.to_datetime(dates_test),
        'signal': signals_test,      # ⭐️ [수정 2] 'signal_prob' -> 'signal'
        'actual_return': returns_test 
    })
    
    df = df.set_index('date').sort_index()
    
    # 5. 롱숏 포트폴리오 백테스트 실행
    print("\n--- [WSTR] 롱숏 포트폴리오 백테스트 ---")
    
    daily_groups = df.groupby(df.index)
    strategy_returns = [] 
    
    from tqdm import tqdm
    for date, group in tqdm(daily_groups, desc="WSTR 백테스트 진행 중"):
        
        if date.weekday() != 0: continue
        if len(group) < 10: continue
            
        # 극단값 제거
        q_01 = group['actual_return'].quantile(0.01)
        q_99 = group['actual_return'].quantile(0.99)
        group['actual_return_clipped'] = group['actual_return'].clip(lower=q_01, upper=q_99)
        
        try:
            # ⭐️ [수정 3] 'signal' (WSTR 시그널)로 10분위 계산
            group['decile'] = pd.qcut(group['signal'], 10, labels=False, duplicates='drop')
        except ValueError:
            continue 
            
        # ⭐️ WSTR (단기 반전) 전략:
        # 시그널(과거 5일 수익률)이 가장 낮은 1분위(decile 0)를 매수 (Long)
        # 시그널이 가장 높은 10분위(decile 9)를 공매도 (Short)
        long_return = group[group['decile'] == 0]['actual_return_clipped'].mean()
        short_return = group[group['decile'] == 9]['actual_return_clipped'].mean()
        
        if pd.isna(long_return) or pd.isna(short_return):
            continue

        weekly_strategy_return = long_return - short_return
        strategy_returns.append(pd.Series([weekly_strategy_return], index=[date]))

    # 6. 최종 성과 분석
    print("\n--- [WSTR] 백테스트 결과 ---")
    
    if not strategy_returns:
        print("오류: 벤치마크 수익률이 계산되지 않았습니다.")
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
    
    # 7. 누적 수익률 그래프 시각화
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('WSTR 벤치마크 누적 수익률 (주간, 2001~)')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익 (1$ 기준)')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('cumulative_returns_WSTR.png')
    print("누적 수익률 그래프가 'cumulative_returns_WSTR.png'로 저장되었습니다.")