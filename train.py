import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Input, Conv2D, LeakyReLU, BatchNormalization, 
    MaxPooling2D, Flatten, Dropout, Dense
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# --- 1. 상수 정의 ---
NPZ_FILE = 'data_L5_R5_with_returns_FIXED.npz'  # 이전 단계에서 생성한 NPZ 파일
IMAGE_SHAPE = (32, 15, 1)     # 5-day 이미지 크기 (Height, Width, Channels)
NUM_CLASSES = 2               # 'Up' / 'Down' (2개 클래스)
MODEL_SAVE_PATH = 'cnn_L5_R5_model_FIXED.keras' # 훈련된 모델 저장 경로

# --- 2. 데이터 로드 함수 ---

def load_data(npz_path):
    """
    .npz 파일에서 이미지(X)와 라벨(y)을 로드합니다.
    라벨(y)을 훈련에 적합하도록 원-핫 인코딩(one-hot encoding)합니다.
    """
    print(f"'{npz_path}' 파일에서 데이터 로드 중...")
    
    if not os.path.exists(npz_path):
        print(f"오류: '{npz_path}' 파일을 찾을 수 없습니다.")
        print("데이터 생성 스크립트를 먼저 실행해 주세요.")
        return None, None, None

    with np.load(npz_path, allow_pickle=True) as data:
        images = data['images']
        labels = data['labels']
        dates = data['dates']

    print("데이터 로드 완료.")
    print(f"  이미지 (X) 형태: {images.shape}")
    print(f"  라벨 (y) 형태: {labels.shape}")

    # 라벨을 원-핫 인코딩으로 변환
    # [0, 1, 0, 1] -> [[1, 0], [0, 1], [1, 0], [0, 1]]
    # (논문 340번: Softmax가 'Up'/'Down' 확률을 출력하므로)
    labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)
    print(f"  원-핫 인코딩된 라벨 (y) 형태: {labels_categorical.shape}")

    # 데이터 타입 변경 (메모리 효율성 및 GPU 호환성)
    # 픽셀 값은 0-255이므로 uint8이 효율적
    # 모델에 입력 시 Keras가 자동으로 0-1 사이 float로 정규화
    images = images.astype('uint8') 
    labels_categorical = labels_categorical.astype('uint8')
    
    return images, labels_categorical, dates

# --- 3. CNN 모델 구축 함수 ---

def build_model(input_shape, num_classes):
    """
    논문의 Figure 3 (5-day image) 아키텍처에 기반한 CNN 모델을 구축합니다.
    """
    print("CNN 모델 구축 시작...")
    
    model = Sequential(name="Stock_CNN_L5")

    # 입력 레이어
    model.add(Input(shape=input_shape))

    # --- Block 1 (논문 Figure 3 참조) ---
    # 5x3 conv, 64
    model.add(Conv2D(64, (5, 3), padding='same', name='conv1'))
    # Batch Normalization (논문 442번)
    model.add(BatchNormalization(name='bn1'))
    # LReLU (논문 1395번, k=0.01)
    model.add(LeakyReLU(alpha=0.01, name='lrelu1'))
    # 2x1 Max-Pool
    model.add(MaxPooling2D(pool_size=(2, 1), name='pool1'))

    # --- Block 2 (논문 Figure 3 참조) ---
    # 5x3 conv, 128
    model.add(Conv2D(128, (5, 3), padding='same', name='conv2'))
    # Batch Normalization (논문 442번)
    model.add(BatchNormalization(name='bn2'))
    # LReLU (논문 1395번, k=0.01)
    model.add(LeakyReLU(alpha=0.01, name='lrelu2'))
    # 2x1 Max-Pool
    model.add(MaxPooling2D(pool_size=(2, 1), name='pool2'))

    # --- FC Head (Fully Connected Head) ---
    # Flatten (Pool2의 출력: 8 x 15 x 128 = 15360)
    # (논문 Figure 3의 'FC 15360'은 Flatten된 크기를 의미)
    model.add(Flatten(name='flatten'))
    
    # Dropout (논문 443번: 50% Dropout)
    model.add(Dropout(0.5, name='dropout'))

    # Output Layer (논문 275번, 340번)
    # 'Up', 'Down' 2개 클래스에 대한 확률을 Softmax로 출력
    model.add(Dense(num_classes, activation='softmax', name='output'))

    print("모델 구축 완료.")
    return model

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    
    # 1. 데이터 로드
    X_data, y_data, dates = load_data(NPZ_FILE)
    
    if X_data is None:
        exit() # 데이터 로드 실패 시 종료

    # --- 5. 논문 기준 훈련/테스트 데이터 분할 ---
    # 논문 은 1993-2000 데이터를 훈련/검증(Training/Validation)으로,
    # 2001-2019 데이터를 테스트(Test)로 사용했습니다.
    
    print("\n--- 데이터셋 분할 (시계열 기준) ---")
    
    # 'dates' 배열을 NumPy datetime 객체로 변환 (비교를 위해)
    # data_L5_R5.npz 생성 시 저장된 날짜 형식을 'datetime64[D]'로 통일
    try:
        dates_np = dates.astype('datetime64[D]')
    except ValueError:
        print("오류: 'dates' 배열을 날짜로 변환할 수 없습니다. .npz 파일을 확인하세요.")
        exit()

    # 논문 기준 분할 날짜 
    # 2001년 1월 1일 이전 데이터는 훈련/검증용
    split_date = np.datetime64('2001-01-01')

    # 마스크(mask) 생성
    train_val_mask = (dates_np < split_date)
    test_mask = (dates_np >= split_date)

    # 마스크를 적용하여 데이터 분할
    X_train_val = X_data[train_val_mask]
    y_train_val = y_data[train_val_mask]
    
    X_test = X_data[test_mask]
    y_test = y_data[test_mask]

    print(f"데이터 분할 완료:")
    print(f"  -> 훈련/검증 (1993-2000): {X_train_val.shape[0]}개 샘플")
    print(f"  -> 테스트 (2001-): {X_test.shape[0]}개 샘플")

    # 메모리 절약을 위해 원본 대용량 데이터 삭제
    import gc
    del X_data, y_data, dates, train_val_mask, test_mask
    gc.collect()

    # 2. 모델 구축
    model = build_model(IMAGE_SHAPE, NUM_CLASSES)

    # 3. 모델 컴파일 (논문 434, 442번 참조)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # 4. 모델 아키텍처 요약 출력
    model.summary()

    # --- 6. 모델 훈련 (1993-2000 데이터 사용) ---
    print("\n--- 모델 훈련 시작 (1993-2000 데이터) ---")
    
    # 콜백 함수 정의
    early_stopper = EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        verbose=1,
        restore_best_weights=True # 가장 좋았던 시점의 가중치로 복원 [cite: 444]
    )
    model_checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 훈련 실행: 1993-2000 데이터(X_train_val)를 훈련에 사용
    # 논문 에 따라 이 1993-2000 데이터를 다시 7:3으로 랜덤 분할하여
    # 70%는 훈련(train)에, 30%는 검증(validation)에 사용
    history = model.fit(
        X_train_val, 
        y_train_val,
        batch_size=128,       # 논문 442번
        epochs=50,            # (EarlyStopping이 알아서 중단시킬 것)
        validation_split=0.3, # 훈련(70%)/검증(30%) 분할 
        callbacks=[early_stopper, model_checkpoint],
        shuffle=True          # 훈련/검증 셋 내부는 랜덤 셔플 
    )

    print("\n훈련 완료!")
    print(f"최상의 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")

    # --- 7. 테스트셋으로 최종 모델 평가 (Out-of-Sample Test) ---
    print("\n--- Out-of-Sample 테스트 시작 (2001- 데이터) ---")
    print(f"훈련된 최상 모델({MODEL_SAVE_PATH})로 테스트셋 평가 중...")
    
    # EarlyStopping의 'restore_best_weights=True' 옵션 덕분에
    # 'model' 객체는 이미 검증 손실이 가장 낮았던 시점의 가중치를 가지고 있음
    # (또는 model.load_weights(MODEL_SAVE_PATH)로 다시 불러와도 됩니다)
    
    test_loss, test_accuracy = model.evaluate(
        X_test, 
        y_test, 
        batch_size=128
    )
    print(f"테스트 데이터 (2001-) 손실 (Loss): {test_loss:.4f}")
    print(f"테스트 데이터 (2001-) 정확도 (Accuracy): {test_accuracy*100:.2f} %")