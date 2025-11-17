import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# --- 1. 상수 정의 ---
NPZ_FILE = 'data_L5_R5_with_returns.npz' # 수익률이 포함된 새 NPZ 파일
MODEL_FILE = 'cnn_L5_R5_model.keras'      # 훈련된 모델 파일
IMAGE_SHAPE = (32, 15, 1)
NUM_CLASSES = 2

# --- 2. 데이터 로드 함수 (수익률 포함) ---
def load_data_for_backtest(npz_path):
    print(f"'{npz_path}' 파일에서 백테스트 데이터 로드 중...")
    if not os.path.exists(npz_path):
        print(f"오류: '{npz_path}' 파일을 찾을 수 없습니다.")
        return None, None, None
        
    with np.load(npz_path, allow_pickle=True) as data:
        images = data['images']
        dates = data['dates']
        returns = data['returns'] # 'labels' 대신 'returns'를 로드

    print("데이터 로드 완료.")
    print(f"  이미지 (X) 형태: {images.shape}")
    print(f"  날짜 (dates) 형태: {dates.shape}")
    print(f"  수익률 (returns) 형태: {returns.shape}")
    
    images = images.astype('uint8')
    returns = returns.astype('float32')
    
    return images, dates, returns

# --- 3. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. 백테스트 데이터 로드
    X_data, dates_data, returns_data = load_data_for_backtest(NPZ_FILE)
    if X_data is None:
        exit()

    # 2. 훈련/테스트 데이터 분할 (train_model.py와 동일한 로직)
    print("\n--- 테스트 데이터셋 분할 (시계열 기준) ---")
    try:
        dates_np = dates_data.astype('datetime64[D]')
    except ValueError:
        print("오류: 'dates' 배열을 날짜로 변환할 수 없습니다.")
        exit()

    split_date = np.datetime64('2001-01-01')
    test_mask = (dates_np >= split_date)

    # 테스트에 필요한 데이터만 추출
    X_test = X_data[test_mask]
    dates_test = dates_data[test_mask]
    returns_test = returns_data[test_mask] # ⭐️ 실제 수익률 테스트셋

    print(f"테스트 (2001-) 데이터: {X_test.shape[0]}개 샘플")

    # 메모리 절약
    del X_data, dates_data, returns_data, test_mask
    gc.collect()

    # 3. 훈련된 모델 로드
    print(f"\n--- 훈련된 모델 ({MODEL_FILE}) 로드 중 ---")
    if not os.path.exists(MODEL_FILE):
        print(f"오류: '{MODEL_FILE}'을 찾을 수 없습니다. 먼저 훈련시켜 주세요.")
        exit()
    model = tf.keras.models.load_model(MODEL_FILE)
    print("모델 로드 완료.")

    # 4. 테스트셋에 대한 'Up' 확률 예측
    # 🚨 이 작업은 1700만 개+ 데이터에 대해 실행되므로 시간이 걸립니다!
    print(f"\n--- 테스트셋 ({X_test.shape[0]}개) 예측 시작 ---")
    # model.predict()는 [P(Down), P(Up)]을 반환
    # 우리는 'Up' 확률인 두 번째 값 [:, 1]이 필요
    predictions = model.predict(X_test, batch_size=1024, verbose=1)
    up_probabilities = predictions[:, 1]
    print("예측 완료.")

    # 5. 백테스트용 데이터프레임 생성
    print("\n--- 백테스트용 데이터프레임 생성 ---")
    df = pd.DataFrame({
        'date': pd.to_datetime(dates_test), # 날짜를 datetime 객체로
        'signal_prob': up_probabilities,   # 모델이 예측한 'Up' 확률
        'actual_return': returns_test      # 해당 주식의 5일 뒤 실제 수익률
    })
    
    # 날짜별로 그룹화할 수 있도록 날짜를 인덱스로 설정
    df = df.set_index('date').sort_index()
    print(df.head())

# 6. 롱숏(Long-Short) 포트폴리오 백테스트 실행
    print("\n--- 롱숏 포트폴리오 백테스트 (주간 리밸런싱 + 극단값 제거) ---")
    
    daily_groups = df.groupby(df.index)
    strategy_returns = [] 
    
    from tqdm import tqdm
    for date, group in tqdm(daily_groups, desc="백테스트 진행 중"):
        
        # 1. 주간 리밸런싱 (월요일에만 실행)
        if date.weekday() != 0:
            continue
            
        if len(group) < 10:
            continue
            
        # ⭐️ [신규 수정] ⭐️
        # Winsorization (극단값 제거)
        # 5일 수익률의 극단적인 오류값을 제거합니다.
        # 매일 상위 1%(q_99)와 하위 1%(q_01)를 초과하는 값은
        # 각각 99%와 1%의 값으로 '캡(cap)'을 씌웁니다.
        q_01 = group['actual_return'].quantile(0.01)
        q_99 = group['actual_return'].quantile(0.99)
        
        # q_01보다 작으면 q_01로, q_99보다 크면 q_99로 값을 제한
        group['actual_return_clipped'] = group['actual_return'].clip(lower=q_01, upper=q_99)
        
        # 2. 10분위 계산 (시그널 기준)
        try:
            # 시그널(prob)을 기준으로 주식을 줄 세웁니다.
            group['decile'] = pd.qcut(group['signal_prob'], 10, labels=False, duplicates='drop')
        except ValueError:
            continue 

        # 3. 포트폴리오 수익률 계산 (클린 데이터 기준)
        # ⭐️ [신규 수정] ⭐️
        # 'actual_return' 대신 깨끗해진 'actual_return_clipped'의 평균을 사용
        long_return = group[group['decile'] == 9]['actual_return_clipped'].mean()
        short_return = group[group['decile'] == 0]['actual_return_clipped'].mean()
        
        if pd.isna(long_return) or pd.isna(short_return):
            continue

        weekly_strategy_return = long_return - short_return
        strategy_returns.append(pd.Series([weekly_strategy_return], index=[date]))

    # 7. 최종 성과 분석 (이 부분은 동일합니다)
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
    
    # 8. 누적 수익률 그래프 시각화 (이 부분은 동일합니다)
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('CNN 롱숏 포트폴리오 누적 수익률 (주간, 극단값 제거, 2001~)')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익 (1$ 기준)')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('cumulative_returns_L5_R5.png')
    print("누적 수익률 그래프가 'cumulative_returns_L5_R5.png'로 저장되었습니다.")