# CNN 주식 예측 시스템 실행 가이드

이 문서는 CNN 기반 주식 예측 시스템의 전체 워크플로우와 각 스크립트의 역할을 설명합니다.

## 📋 목차

1. [전체 워크플로우 개요](#전체-워크플로우-개요)
2. [CNN 파이프라인](#cnn-파이프라인)
3. [WSTR 벤치마크 및 비교](#wstr-벤치마크-및-비교)
4. [실행 순서 요약](#실행-순서-요약)

---

## 전체 워크플로우 개요

이 시스템은 두 가지 주요 파이프라인으로 구성됩니다:

1. **CNN 파이프라인**: 주식 차트 이미지를 CNN으로 학습하여 주가 상승/하락을 예측
2. **WSTR 벤치마크 파이프라인**: 과거 수익률 기반 벤치마크 전략과 CNN 전략을 비교

```
CNN 파이프라인:
make_npz.py → train.py → run_predictions.py

WSTR 벤치마크 파이프라인:
make_wstr_npz.py → cnn_wstr_compare.py
```

---

## CNN 파이프라인

### 1. `make_npz.py` - 데이터 전처리 및 이미지 생성

**목적**: 주식 CSV 파일들을 읽어서 CNN 학습용 차트 이미지와 라벨을 생성하여 NPZ 파일로 저장

#### 주요 기능

1. **데이터 전처리** (`preprocess_data`)
   - CSV 파일에서 날짜, OHLCV 데이터 로드
   - 1993-01-01 이후 데이터만 사용
   - 조정 수익률(AdjReturn) 계산
   - O/H/L 가격을 종가 대비 비율로 변환

2. **차트 이미지 생성** (`generate_image_from_window`)
   - 롤링 윈도우 방식으로 n일치 데이터를 차트 이미지로 변환
   - 논문 기반 스케일링 로직 적용
   - 가격 영역(4/5)과 거래량 영역(1/5)으로 분리
   - High-Low 바, Open/Close 점, 이동평균선 포함
   - 이미지 크기:
     - 5일: 32x15 픽셀
     - 20일: 64x60 픽셀
     - 60일: 96x180 픽셀

3. **라벨 및 수익률 계산** (`calculate_label_and_return`)
   - 미래 n일 누적 수익률 계산
   - 수익률 > 0% → 라벨 1 (Up)
   - 수익률 ≤ 0% → 라벨 0 (Down)
   - 실제 수익률 값도 함께 저장

4. **멀티프로세싱 처리** (`process_all_files`)
   - 모든 CSV 파일을 병렬 처리
   - 결과를 하나의 NPZ 파일로 통합

#### 입력
- `stocks_folder`: 주식 CSV 파일들이 있는 폴더 경로
- `n_days`: 이미지 기간 (5, 20, 60 중 선택)

#### 출력
- `data_L5_R5_with_returns.npz`: 다음 데이터 포함
  - `images`: 차트 이미지 배열 (N, H, W, 1)
  - `labels`: 라벨 배열 (0 또는 1)
  - `dates`: 날짜 배열
  - `returns`: 실제 수익률 배열

#### 실행 예시
```python
process_all_files(
    stocks_folder='nasdaq_yfinance_20200401/stocks',
    output_file='data_L5_R5_with_returns.npz',
    n_days=5
)
```

---

### 2. `train.py` - CNN 모델 훈련

**목적**: 생성된 NPZ 파일을 사용하여 CNN 모델을 훈련하고 저장

#### 주요 기능

1. **데이터 로드** (`load_data`)
   - NPZ 파일에서 이미지, 라벨, 날짜 로드
   - 라벨을 원-핫 인코딩으로 변환 (2개 클래스: Down, Up)
   - 데이터 타입 최적화 (uint8)

2. **시계열 데이터 분할**
   - 1993-2000: 훈련/검증 데이터
   - 2001-: 테스트 데이터 (Out-of-Sample)
   - 훈련/검증 데이터는 7:3으로 랜덤 분할

3. **CNN 모델 구축** (`build_model`)
   - 논문 Figure 3 기반 아키텍처
   - **Block 1**: Conv2D(64, 5x3) → BatchNorm → LeakyReLU → MaxPool(2x1)
   - **Block 2**: Conv2D(128, 5x3) → BatchNorm → LeakyReLU → MaxPool(2x1)
   - **FC Head**: Flatten → Dropout(0.5) → Dense(2, softmax)
   - 최종 출력: [P(Down), P(Up)] 확률

4. **모델 훈련**
   - Optimizer: Adam (learning_rate=1e-5)
   - Loss: Categorical Crossentropy
   - Batch Size: 128
   - Early Stopping: val_loss 기준, patience=2
   - Model Checkpoint: 최상의 모델 자동 저장

#### 입력
- `data_L5_R5_with_returns.npz`: `make_npz.py`에서 생성한 파일

#### 출력
- `cnn_L5_R5_model.keras`: 훈련된 모델 파일
- 콘솔 출력: 훈련 과정 및 테스트 정확도

#### 주요 설정
```python
NPZ_FILE = 'data_L5_R5_with_returns.npz'
IMAGE_SHAPE = (32, 15, 1)  # 5-day 이미지
NUM_CLASSES = 2
MODEL_SAVE_PATH = 'cnn_L5_R5_model.keras'
```

---

### 3. `run_predictions.py` - 예측 수행 및 결과 저장

**목적**: 훈련된 모델로 테스트 데이터에 대한 예측을 수행하고 백테스트용 데이터 생성

#### 주요 기능

1. **테스트 데이터 로드**
   - NPZ 파일에서 이미지, 날짜, 실제 수익률 로드
   - 2001년 이후 데이터만 필터링 (테스트 기간)

2. **모델 로드**
   - `cnn_L5_R5_model.keras` 파일 로드

3. **예측 수행**
   - 테스트셋 전체에 대해 'Up' 확률 예측
   - Batch size: 1024 (대용량 데이터 처리 최적화)

4. **백테스트용 데이터프레임 생성**
   - 날짜, 예측 확률(signal_prob), 실제 수익률(actual_return) 포함
   - Parquet 형식으로 저장 (압축 및 빠른 읽기)

#### 입력
- `data_L5_R5_with_returns.npz`: 원본 데이터
- `cnn_L5_R5_model.keras`: 훈련된 모델

#### 출력
- `backtest_data_with_predictions.parquet`: 다음 컬럼 포함
  - `date`: 날짜 (인덱스)
  - `signal_prob`: CNN 예측 'Up' 확률 (0~1)
  - `actual_return`: 실제 미래 수익률

#### 실행 후
이 파일 실행 후 `run_backtest.py`를 실행하여 롱숏 포트폴리오 백테스트를 수행할 수 있습니다.

---

## WSTR 벤치마크 및 비교

### 4. `make_wstr_npz.py` - WSTR 벤치마크 데이터 생성

**목적**: 과거 수익률 기반 WSTR(Weighted Short-Term Return) 벤치마크 데이터 생성

#### 주요 기능

1. **데이터 전처리** (`preprocess_data`)
   - CSV 파일에서 일간 수익률(AdjReturn)만 추출
   - 1993-01-01 이후 데이터만 사용

2. **WSTR 시그널 계산** (`process_single_file`)
   - 과거 n일 누적 수익률 계산 (T-5 ~ T-1)
   - 이 값이 WSTR 시그널 (단기 모멘텀 지표)

3. **실제 수익률 계산**
   - 미래 n일 누적 수익률 계산 (T+1 ~ T+5)
   - 벤치마크 전략의 성과 평가에 사용

4. **멀티프로세싱 처리**
   - 모든 CSV 파일을 병렬 처리하여 NPZ 파일로 저장

#### 입력
- `stocks_folder`: 주식 CSV 파일들이 있는 폴더 경로
- `n_days`: 수익률 계산 기간 (기본값: 5)

#### 출력
- `benchmark_data_WSTR.npz`: 다음 데이터 포함
  - `dates`: 날짜 배열
  - `signals`: WSTR 시그널 (과거 n일 누적 수익률)
  - `returns`: 실제 미래 수익률

#### WSTR 전략 설명
- **시그널**: 과거 5일 누적 수익률 (단기 모멘텀)
- **전략 타입**: 반전(Reversal) 전략
  - 낮은 시그널(Decile 0) → 매수 (과거 하락 → 반등 기대)
  - 높은 시그널(Decile 9) → 매도 (과거 상승 → 하락 기대)

---

### 5. `cnn_wstr_compare.py` - CNN vs WSTR 성과 비교

**목적**: CNN 전략과 WSTR 벤치마크 전략의 성과를 비교 분석

#### 주요 기능

1. **데이터 로드**
   - CNN 예측 데이터: `backtest_data_with_predictions.parquet`
   - WSTR 벤치마크 데이터: `benchmark_data_WSTR.npz`
   - 둘 다 2001년 이후 테스트 기간만 사용

2. **롱숏 포트폴리오 백테스트** (`run_backtest`)
   - **CNN 전략** (모멘텀):
     - 시그널 기준 십분위(Decile) 분할
     - High(Decile 9) 매수, Low(Decile 0) 매도
   - **WSTR 전략** (반전):
     - 시그널 기준 십분위 분할
     - Low(Decile 0) 매수, High(Decile 9) 매도
   - 주간 리밸런싱 (월요일만)
   - Winsorization: 극단값 제거 (1%, 99% quantile)

3. **성과 분석** (`calculate_performance`)
   - 주간 평균 수익률
   - 주간 수익률 변동성
   - 연간 샤프 비율 (Annualized Sharpe Ratio)
   - 누적 수익률 계산

4. **시각화**
   - CNN과 WSTR의 누적 수익률 비교 그래프
   - 로그 스케일로 표시
   - 각 전략의 샤프 비율 포함

#### 입력
- `backtest_data_with_predictions.parquet`: CNN 예측 결과
- `benchmark_data_WSTR.npz`: WSTR 벤치마크 데이터

#### 출력
- `cumulative_returns_COMPARISON.png`: 비교 그래프
- 콘솔 출력: 각 전략의 성과 지표

#### 전략 비교 요약

| 전략 | 시그널 | 전략 타입 | 매수/매도 기준 |
|------|--------|-----------|----------------|
| CNN | CNN 예측 'Up' 확률 | 모멘텀 | High(9) 매수, Low(0) 매도 |
| WSTR | 과거 5일 누적 수익률 | 반전 | Low(0) 매수, High(9) 매도 |

---

## 실행 순서 요약

### 1단계: CNN 파이프라인

```bash
# 1. 데이터 전처리 및 이미지 생성
python make_npz.py
# 출력: data_L5_R5_with_returns.npz

# 2. CNN 모델 훈련
python train.py
# 출력: cnn_L5_R5_model.keras

# 3. 예측 수행
python run_predictions.py
# 출력: backtest_data_with_predictions.parquet
```

### 2단계: WSTR 벤치마크 및 비교

```bash
# 4. WSTR 벤치마크 데이터 생성
python make_wstr_npz.py
# 출력: benchmark_data_WSTR.npz

# 5. CNN vs WSTR 성과 비교
python cnn_wstr_compare.py
# 출력: cumulative_returns_COMPARISON.png
```

### 선택 사항: 개별 CNN 백테스트

```bash
# CNN 전략만 개별적으로 백테스트 (선택 사항)
python run_backtest.py
# 출력: cumulative_returns_L5_R5.png
```

---

## 주요 파일 설명

### 입력 파일
- `nasdaq_yfinance_20200401/stocks/**/*.csv`: 주식 OHLCV 데이터 (yfinance 형식)

### 중간 파일
- `data_L5_R5_with_returns.npz`: CNN 학습용 이미지 및 라벨
- `benchmark_data_WSTR.npz`: WSTR 벤치마크 데이터
- `cnn_L5_R5_model.keras`: 훈련된 CNN 모델
- `backtest_data_with_predictions.parquet`: CNN 예측 결과

### 출력 파일
- `cumulative_returns_COMPARISON.png`: CNN vs WSTR 비교 그래프
- `cumulative_returns_L5_R5.png`: CNN 개별 백테스트 그래프 (선택)

---

## 데이터 분할 기준

- **훈련/검증**: 1993-01-01 ~ 2000-12-31
  - 내부적으로 7:3 랜덤 분할 (훈련 70%, 검증 30%)
- **테스트**: 2001-01-01 ~ (데이터 끝까지)
  - Out-of-Sample 테스트 (실제 성과 평가)

---

## 주요 파라미터

### 이미지 생성 (`make_npz.py`)
- `n_days`: 5, 20, 60 중 선택
- 이미지 크기: 자동 계산 (n_days * 3px 너비)

### 모델 훈련 (`train.py`)
- Learning Rate: 1e-5
- Batch Size: 128
- Epochs: 50 (Early Stopping으로 조기 종료 가능)
- Dropout: 0.5

### 백테스트 (`cnn_wstr_compare.py`)
- 리밸런싱 주기: 주 1회 (월요일)
- 십분위 분할: 10개 그룹
- Winsorization: 1%, 99% quantile

---

## 주의사항

1. **메모리 사용량**: 대용량 데이터 처리 시 충분한 RAM 필요
2. **실행 시간**: 
   - `make_npz.py`: 멀티프로세싱 사용, 수 시간 소요 가능
   - `train.py`: GPU 사용 시 빠름, CPU만 사용 시 수 시간 소요
   - `run_predictions.py`: 대용량 테스트셋 예측 시 시간 소요
3. **파일 의존성**: 각 스크립트는 이전 단계의 출력 파일이 필요
4. **데이터 경로**: `stocks_folder` 경로를 실제 환경에 맞게 수정 필요

---

## 문제 해결

### NPZ 파일을 찾을 수 없음
- `make_npz.py`를 먼저 실행했는지 확인
- 파일 경로가 올바른지 확인

### 모델 파일을 찾을 수 없음
- `train.py`를 먼저 실행했는지 확인
- 모델 훈련이 완료되었는지 확인

### 메모리 부족 오류
- 배치 크기 줄이기
- 멀티프로세싱 워커 수 줄이기 (`num_workers` 파라미터)
- 데이터를 더 작은 단위로 분할

---

## 참고

이 시스템은 논문 "Image-based Stock Price Prediction using CNN"의 구현을 기반으로 합니다.
- 차트 이미지 생성 로직은 논문의 스케일링 방법을 따릅니다
- CNN 아키텍처는 논문 Figure 3을 참조합니다
- 데이터 분할 및 백테스트 방법은 논문의 실험 설계를 따릅니다

