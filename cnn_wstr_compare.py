import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

# --- 0. 기본 설정 ---

# 한글 폰트 설정 (macOS 사용 중이신 것 같아 AppleGothic으로 설정합니다)
# Windows 사용 시: 'Malgun Gothic'
# Linux 사용 시: 'NanumGothic' (설치 필요)
try:
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    print("AppleGothic 폰트를 찾을 수 없습니다. 기본 폰트로 설정됩니다.")
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 상수 정의 ---
CNN_PREDICTIONS_FILE = 'backtest_data_with_predictions.parquet'
WSTR_DATA_FILE = 'benchmark_data_WSTR.npz'
SPLIT_DATE = np.datetime64('2001-01-01')
OUTPUT_CHART_FILE = 'cumulative_returns_COMPARISON.png'

# --- 2. 데이터 로드 함수 (수정됨) ---

def load_wstr_data(npz_path):
    """
    WSTR 벤치마크 데이터(NPZ)를 로드하여 DataFrame으로 반환합니다.
    """
    print(f"'{npz_path}' 파일에서 WSTR 벤치마크 데이터 로드 중...")
    if not os.path.exists(npz_path):
        print(f"오류: '{npz_path}' 파일을 찾을 수 없습니다.")
        return None
        
    with np.load(npz_path, allow_pickle=True) as data:
        dates = data['dates']
        signals = data['signals'] # WSTR 시그널
        returns = data['returns'] # WSTR의 실제 미래 수익률
    
    df_wstr = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'wstr_signal': signals,
        'actual_return': returns
    })
    
    # 테스트 기간 필터링
    df_wstr = df_wstr[df_wstr['date'] >= SPLIT_DATE].copy()
    
    # 날짜별로 그룹화할 수 있도록 날짜를 인덱스로 설정
    df_wstr = df_wstr.set_index('date').sort_index()
    
    print(f"WSTR 데이터 로드 완료. {len(df_wstr)}개 샘플 (2001-)")
    return df_wstr

def load_cnn_data(parquet_path):
    """
    CNN 예측 데이터(Parquet)를 로드하여 DataFrame으로 반환합니다.
    """
    print(f"'{parquet_path}' 파일에서 CNN 예측 데이터 로드 중...")
    if not os.path.exists(parquet_path):
        print(f"오류: '{parquet_path}' 파일을 찾을 수 없습니다.")
        return None
    
    df_cnn = pd.read_parquet(parquet_path)
    
    # 컬럼 이름 변경 (WSTR과 겹치지 않게)
    df_cnn = df_cnn.rename(columns={'signal_prob': 'cnn_signal'})
    
    # Parquet 파일은 이미 1번 스크립트에서 2001년 이후로 필터링되었습니다.
    print(f"CNN 데이터 로드 완료. {len(df_cnn)}개 샘플 (2001-)")
    return df_cnn

# --- 3. 공통 백테스트 함수 (수정됨) ---

def run_backtest(df_data, signal_column, strategy_name, strategy_type='momentum'):
    """
    주어진 DataFrame과 시그널 컬럼으로 롱숏 백테스트를 수행합니다.
    
    :param df_data: 'actual_return'과 'signal_column'이 포함된 DataFrame (날짜 인덱스)
    :param signal_column: 십분위 분석에 사용할 시그널 컬럼 이름
    :param strategy_name: 로그 출력용 전략 이름
    :param strategy_type: 'momentum' (High-Low) 또는 'reversal' (Low-High)
    :return: 주간 수익률 Pandas Series
    """
    
    print(f"\n--- [{strategy_name}] 롱숏 포트폴리오 백테스트 시작 ---")
    
    # 날짜별로 그룹화
    daily_groups = df_data.groupby(df_data.index) 
    
    strategy_returns = [] 
    
    for date, group in tqdm(daily_groups, desc=f"[{strategy_name}] 백테스트 진행 중"):
        
        # 논문과 동일하게 주 1회 (월요일) 리밸런싱
        if date.weekday() != 0: 
            continue
            
        # 포트폴리오 구성에 필요한 최소 주식 수 (너무 적으면 십분위 분할 불가)
        if len(group) < 20: 
            continue
            
        # 극단값 제거 (Winsorization)
        q_01 = group['actual_return'].quantile(0.01)
        q_99 = group['actual_return'].quantile(0.99)
        group['return_clipped'] = group['actual_return'].clip(lower=q_01, upper=q_99)
        
        try:
            # 시그널을 기준으로 십분위(Decile) 분할
            group['decile'] = pd.qcut(group[signal_column], 10, labels=False, duplicates='drop')
        except ValueError:
            # 가끔 동일한 시그널 값이 많아 십분위 분할이 안될 수 있음
            continue 
            
        # 십분위별 평균 수익률 계산
        decile_returns = group.groupby('decile')['return_clipped'].mean()
        
        if strategy_type == 'momentum':
            # CNN: High(9) 매수, Low(0) 매도
            long_return = decile_returns.get(9, np.nan)
            short_return = decile_returns.get(0, np.nan)
        elif strategy_type == 'reversal':
            # WSTR: Low(0) 매수, High(9) 매도 (단기 반전 전략)
            long_return = decile_returns.get(0, np.nan)
            short_return = decile_returns.get(9, np.nan)
        
        # 롱/숏 포지션이 모두 존재할 경우에만 수익률 계산
        if not pd.isna(long_return) and not pd.isna(short_return):
            weekly_return = long_return - short_return
            strategy_returns.append(pd.Series([weekly_return], index=[date]))

    return pd.concat(strategy_returns)

# --- 4. 성과 분석 헬퍼 함수 (원본과 동일) ---
def calculate_performance(weekly_returns_series, strategy_name):
    print(f"\n--- [{strategy_name}] 백테스트 결과 ---")
    if weekly_returns_series.empty:
        print("오류: 수익률이 계산되지 않았습니다.")
        return 0, pd.Series()

    weekly_returns_series = weekly_returns_series.dropna()
    mean_weekly_return = weekly_returns_series.mean()
    std_weekly_return = weekly_returns_series.std()
    
    annualized_sharpe_ratio = 0.0
    if std_weekly_return > 0:
        # 연간 샤프 비율 = (주간 평균수익 / 주간 변동성) * sqrt(52주)
        annualized_sharpe_ratio = (mean_weekly_return / std_weekly_return) * np.sqrt(52)

    print(f"총 주간 수익률 평균: {mean_weekly_return*100:.4f} %")
    print(f"총 주간 수익률 변동성: {std_weekly_return*100:.4f} %")
    print(f"연간 샤프 비율 (Annualized Sharpe Ratio): {annualized_sharpe_ratio:.4f}")
    
    # 누적 수익률 계산 (그래프용)
    clipped_weekly_returns = weekly_returns_series.clip(lower=-0.99) # 한 주에 -100% 이상 손실 방지
    cumulative_returns = (1 + clipped_weekly_returns).cumprod()
    
    return annualized_sharpe_ratio, cumulative_returns

# --- 5. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. CNN 데이터 로드 (병합 X)
    df_cnn = load_cnn_data(CNN_PREDICTIONS_FILE)
    if df_cnn is None:
        exit()
        
    # 2. WSTR 데이터 로드 (병합 X)
    df_wstr = load_wstr_data(WSTR_DATA_FILE)
    if df_wstr is None:
        exit()

    # --- 3. [핵심] 백테스트 별도 실행 ---
    
    # CNN 전략 백테스트 (모멘텀: High-Low)
    cnn_returns_series = run_backtest(
        df_cnn, 
        'cnn_signal', 
        "CNN (L5/R5)", 
        strategy_type='momentum'
    )
    
    # WSTR 전략 백테스트 (반전: Low-High)
    wstr_returns_series = run_backtest(
        df_wstr, 
        'wstr_signal', 
        "WSTR Benchmark", 
        strategy_type='reversal'
    )
    
    del df_cnn, df_wstr
    gc.collect()

    # --- 4. 최종 성과 분석 ---
    cnn_sharpe, cnn_cum_returns = calculate_performance(cnn_returns_series, "CNN (L5/R5)")
    wstr_sharpe, wstr_cum_returns = calculate_performance(wstr_returns_series, "WSTR Benchmark")
    
    # --- 5. 누적 수익률 비교 그래프 시각화 (원본과 동일) ---
    print(f"\n--- 비교 그래프 저장 중 -> {OUTPUT_CHART_FILE} ---")
    plt.figure(figsize=(12, 7))
    
    cnn_cum_returns.plot(label=f"CNN (L5/R5) - Sharpe: {cnn_sharpe:.2f}", color='blue', linewidth=2)
    wstr_cum_returns.plot(label=f"WSTR - Sharpe: {wstr_sharpe:.2f}", color='red', linestyle='--')
    
    plt.title('CNN (L5/R5) vs WSTR 벤치마크 누적 수익률 비교 (2001~)')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익 (로그 스케일, 1$ 기준)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.yscale('log') # 논문처럼 로그 스케일로 표시
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_FILE)
    
    print(f"비교 그래프가 '{OUTPUT_CHART_FILE}' 파일로 저장되었습니다.")