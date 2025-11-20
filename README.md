# CNN Stock Prediction Pipeline

주식 차트 이미지를 CNN으로 분석하여 상승/하락을 예측하는 파이프라인입니다.

## 📋 목차

- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [실행 순서](#실행-순서)
- [스크립트 상세 설명](#스크립트-상세-설명)
- [데이터 형식](#데이터-형식)
- [요구사항](#요구사항)

## 개요

이 프로젝트는 논문 기반의 CNN 모델을 사용하여 주식 차트 이미지로부터 향후 5일간의 상승/하락을 예측합니다. 

주요 특징:
- 주식 데이터를 차트 이미지로 변환 (OHLCV 데이터 시각화)
- CNN 모델을 통한 이미지 분류 (Up/Down 예측)
- NASDAQ 주식 필터링 및 Stale stocks 제외
- Long/Short 전략 데이터 생성

## 프로젝트 구조

```
.
├── make_image.py              # 주식 데이터를 이미지로 변환
├── train.py                   # CNN 모델 훈련
├── filter_nasdaq_stocks.py    # NASDAQ 주식 필터링
├── find_stale_stocks.py       # Stale stocks 탐지
├── save_predictions.py        # 예측 확률 저장
├── create_longshort_csv.py    # Long/Short 전략 CSV 생성
├── nasdaq_yfinance_20200401/
│   ├── stocks/                # 개별 주식 CSV 파일들
│   ├── symbols_valid_meta.csv # 원본 심볼 메타데이터
│   ├── nasdaq_stocks_filtered.csv
│   └── nasdaq_stocks_symbols.txt
├── data_L5_R5.npz             # 생성된 이미지 데이터셋
├── cnn_L5_R5_model_final.keras # 훈련된 모델
└── stocks_combined.csv        # 예측 확률이 포함된 주식 데이터
```

## 실행 순서

### 1단계: 데이터 필터링

```bash
# NASDAQ 주식 필터링 (ETF 제외, 주요 거래소만)
python filter_nasdaq_stocks.py
```

**출력 파일:**
- `nasdaq_yfinance_20200401/nasdaq_stocks_filtered.csv`
- `nasdaq_yfinance_20200401/nasdaq_stocks_symbols.txt`

### 2단계: Stale Stocks 탐지

```bash
# Volume=0이고 가격이 변하지 않는 종목 찾기
python find_stale_stocks.py
```

**출력 파일:**
- `stale_stocks_summary.csv` - 종목별 요약 정보
- `stale_stocks_detailed.csv` - 상세 기간 정보
- `stale_stocks_30days_plus.csv` - 30일 이상 연속된 종목 (수동 생성 필요)

### 3단계: 이미지 데이터 생성

```bash
# 주식 CSV 파일들을 차트 이미지로 변환
python make_image.py
```

**출력 파일:**
- `data_L5_R5.npz` - 이미지, 라벨, 날짜, 수익률, 티커 정보가 포함된 NumPy 압축 파일

**주요 설정:**
- `n_days=5`: 5일간의 데이터로 이미지 생성
- 이미지 크기: 32x15 (가격 영역 4/5, 거래량 영역 1/5)
- 멀티프로세싱 지원

### 4단계: 모델 훈련

```bash
# CNN 모델 훈련 (1993-2000 데이터 사용)
python train.py
```

**출력 파일:**
- `cnn_L5_R5_model_final.keras` - 훈련된 모델 가중치

**훈련 설정:**
- 훈련 데이터: 1993-01-01 ~ 2000-12-31
- 테스트 데이터: 2001-01-01 이후
- Early Stopping 적용 (patience=2)
- Validation split: 0.3

### 5단계: 예측 확률 저장

```bash
# 테스트 데이터에 대한 예측 확률을 stocks_combined.csv에 저장
python save_predictions.py
```

**필터링 조건:**
- NASDAQ 티커만 포함
- Stale stocks 제외 (max_consecutive_days >= 60)

**입력 파일:**
- `data_L5_R5.npz`
- `cnn_L5_R5_model_final.keras`
- `nasdaq_yfinance_20200401/nasdaq_stocks_symbols.txt`
- `stale_stocks_30days_plus.csv`
- `nasdaq_yfinance_20200401/stocks_combined.csv`

**출력 파일:**
- `nasdaq_yfinance_20200401/stocks_combined.csv` (업데이트됨)

### 6단계: Long/Short 전략 CSV 생성

```bash
# 날짜별로 probability_up이 가장 높은/낮은 종목 선택
python create_longshort_csv.py
```

**출력 파일:**
- `longshort_2001_after.csv` - 날짜별 Long/Short 종목 정보

## 스크립트 상세 설명

### `make_image.py`

주식 CSV 파일을 차트 이미지로 변환합니다.

**주요 기능:**
- `preprocess_data()`: yfinance CSV 전처리 (조정 수익률 계산, O/H/L factor 계산)
- `generate_image_from_window()`: 롤링 윈도우로 차트 이미지 생성
- `calculate_label_and_return()`: 향후 n일간의 수익률로 라벨 생성 (Up=1, Down=0)
- `process_all_files()`: 멀티프로세싱으로 모든 CSV 파일 처리

**이미지 생성 로직:**
- 상대 가격 정규화 (첫날 종가 = 1.0)
- 가격 영역: High-Low 바, Open/Close 점, 이동평균선
- 거래량 영역: 하단 1/5 영역에 거래량 바

### `train.py`

CNN 모델을 구축하고 훈련합니다.

**모델 아키텍처:**
```
Input (32, 15, 1)
  ↓
Conv2D(64, (5,3)) + BatchNorm + LeakyReLU + MaxPool(2,1)
  ↓
Conv2D(128, (5,3)) + BatchNorm + LeakyReLU + MaxPool(2,1)
  ↓
Flatten + Dropout(0.5)
  ↓
Dense(2, softmax)  # Up/Down 확률
```

**훈련 설정:**
- Optimizer: Adam (lr=1e-5)
- Loss: Categorical Crossentropy
- Batch size: 128
- Epochs: 50 (Early Stopping)

### `filter_nasdaq_stocks.py`

주요 거래소 상장 주식을 필터링합니다.

**필터링 조건:**
- ETF 제외 (`ETF == 'N'`)
- NextShares 제외
- 테스트 이슈 제외 (`Test Issue == 'N'`)
- 허용 거래소: Q (NASDAQ), N (NYSE), A (NYSE American), P (NYSE Arca)
- 문제가 있는 재무 상태 제외 (D, E, H, S 제외)

### `find_stale_stocks.py`

Volume=0이고 가격이 변하지 않는 종목을 탐지합니다.

**탐지 조건:**
- Volume = 0
- Open, High, Low, Close가 전날과 동일
- 1993-01-01 이후 데이터만 검사

**출력 정보:**
- 최대 연속 일수 (`max_consecutive_days`)
- 전체 stale 일수 (`total_stale_days`)
- Stale 기간 리스트

### `save_predictions.py`

훈련된 모델로 테스트 데이터에 대한 예측 확률을 계산하고 저장합니다.

**주요 기능:**
- 테스트 데이터 로드 (2001-01-01 이후)
- NASDAQ 티커 필터링
- Stale stocks 제외 (max_consecutive_days >= 60)
- 예측 확률을 `stocks_combined.csv`에 병합

### `create_longshort_csv.py`

날짜별로 Long/Short 전략 종목을 선택합니다.

**선택 기준:**
- Long: `probability_up`이 가장 높은 종목
- Short: `probability_up`이 가장 낮은 종목
- Volume > 0인 종목만 선택

**출력 컬럼:**
- Date, Long, long_close, long_ret5, long_prob_up
- Short, short_close, short_ret5, short_prob_up

## 데이터 형식

### 입력 CSV 형식 (yfinance)

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-01,100.0,105.0,99.0,103.0,103.0,1000000
...
```

### NPZ 파일 형식 (`data_L5_R5.npz`)

```python
{
    'images': np.array,      # (N, 32, 15, 1) uint8
    'labels': np.array,      # (N,) uint8 (0=Down, 1=Up)
    'dates': np.array,       # (N,) str 'YYYY-MM-DD'
    'returns': np.array,     # (N,) float32 (실제 수익률)
    'tickers': np.array      # (N,) str (티커 심볼)
}
```

### stocks_combined.csv 형식

```csv
ticker,date,Open,High,Low,Close,Adj Close,Volume,ret5,probability_up
AAPL,2020-01-01,100.0,105.0,99.0,103.0,103.0,1000000,0.05,0.65
...
```

## 요구사항

### Python 패키지

```
tensorflow>=2.0
keras>=2.0
numpy
pandas
opencv-python
tqdm
```

### 데이터 파일

1. `nasdaq_yfinance_20200401/symbols_valid_meta.csv` - 원본 심볼 메타데이터
2. `nasdaq_yfinance_20200401/stocks/*.csv` - 개별 주식 CSV 파일들
3. `nasdaq_yfinance_20200401/stocks_combined.csv` - 통합 주식 데이터 (선택사항)

### 하드웨어

- 멀티프로세싱을 위한 다중 CPU 코어 권장
- GPU 사용 시 TensorFlow GPU 버전 설치 필요

## 주의사항

1. **실행 순서 준수**: 스크립트는 순차적으로 실행해야 합니다.
2. **메모리 관리**: 대용량 데이터셋 처리 시 충분한 메모리 필요
3. **Stale stocks 필터**: `stale_stocks_30days_plus.csv`는 `find_stale_stocks.py` 실행 후 수동으로 생성하거나 필터링해야 합니다.
4. **날짜 형식**: 모든 날짜는 'YYYY-MM-DD' 형식을 사용합니다.

## 참고

- 논문 기반 아키텍처 사용
- 시계열 데이터 분할 (1993-2000: 훈련, 2001-: 테스트)
- 멀티프로세싱으로 이미지 생성 속도 향상
- 메모리 효율적인 NPZ 형식 사용
