import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc

# --- 1. 상수 정의 ---
NPZ_FILE = 'data_L5_R5_with_returns_FIXED.npz'
MODEL_FILE = 'cnn_L5_R5_model_FIXED.keras'
PREDICTIONS_FILE = 'backtest_data_with_predictions_FIXED.parquet' # ⭐️ 저장할 파일 이름

# --- 2. 데이터 로드 함수 (원본과 동일) ---
def load_data_for_backtest(npz_path):
    print(f"'{npz_path}' 파일에서 백테스트 데이터 로드 중...")
    if not os.path.exists(npz_path):
        print(f"오류: '{npz_path}' 파일을 찾을 수 없습니다.")
        return None, None, None
        
    with np.load(npz_path, allow_pickle=True) as data:
        images = data['images']
        dates = data['dates']
        returns = data['returns']

    print("데이터 로드 완료.")
    images = images.astype('uint8')
    returns = returns.astype('float32')
    
    return images, dates, returns

# --- 3. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. 백테스트 데이터 로드
    X_data, dates_data, returns_data = load_data_for_backtest(NPZ_FILE)
    if X_data is None:
        exit()

    # 2. 훈련/테스트 데이터 분할 (원본과 동일)
    print("\n--- 테스트 데이터셋 분할 (시계열 기준) ---")
    try:
        dates_np = dates_data.astype('datetime64[D]')
    except ValueError:
        print("오류: 'dates' 배열을 날짜로 변환할 수 없습니다.")
        exit()

    split_date = np.datetime64('2001-01-01')
    test_mask = (dates_np >= split_date)

    X_test = X_data[test_mask]
    dates_test = dates_data[test_mask]
    returns_test = returns_data[test_mask] 

    print(f"테스트 (2001-) 데이터: {X_test.shape[0]}개 샘플")
    del X_data, dates_data, returns_data, test_mask
    gc.collect()

    # 3. 훈련된 모델 로드 (원본과 동일)
    print(f"\n--- 훈련된 모델 ({MODEL_FILE}) 로드 중 ---")
    if not os.path.exists(MODEL_FILE):
        print(f"오류: '{MODEL_FILE}'을 찾을 수 없습니다. 먼저 훈련시켜 주세요.")
        exit()
    model = tf.keras.models.load_model(MODEL_FILE)
    print("모델 로드 완료.")

    # 4. 테스트셋에 대한 'Up' 확률 예측 (원본과 동일)
    print(f"\n--- 테스트셋 ({X_test.shape[0]}개) 예측 시작 (시간이 오래 걸립니다) ---")
    predictions = model.predict(X_test, batch_size=1024, verbose=1)
    up_probabilities = predictions[:, 1]
    print("예측 완료.")
    
    # ⭐️ 예측에 사용된 X_test는 이제 필요 없으므로 메모리에서 삭제
    del X_test, predictions, model
    gc.collect()

    # 5. 백테스트용 데이터프레임 생성 (원본과 동일)
    print("\n--- 백테스트용 데이터프레임 생성 ---")
    df = pd.DataFrame({
        'date': dates_test,
        'signal_prob': up_probabilities,
        'actual_return': returns_test
    })
    
    # 날짜별로 그룹화할 수 있도록 날짜를 인덱스로 설정
    df = df.set_index('date').sort_index()
    print(df.head())

    # ⭐️⭐️⭐️ [핵심] 5번 과정이 끝난 후, 결과를 파일로 저장 ⭐️⭐️⭐️
    print(f"\n--- 예측 결과 {PREDICTIONS_FILE} 파일로 저장 중 ---")
    df.to_parquet(PREDICTIONS_FILE, compression='gzip')
    print("저장 완료. 이제부터 run_backtest.py를 실행하세요.")