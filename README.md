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

**데이터 전처리 상세 (`preprocess_data()`):**

1. **날짜 처리**
   - `Date` 컬럼 존재 여부 확인
   - `Date` 컬럼을 `pd.to_datetime()`으로 변환
   - 날짜를 인덱스로 설정하고 오름차순 정렬

2. **데이터 필터링**
   - 1993-01-01 이전 데이터 제거 (논문 기준 시작 날짜)
   - 최소 데이터 길이 확인: 10일 미만인 경우 제외

3. **조정 수익률 계산**
   - `AdjReturn = Adj Close.pct_change()`: 조정 종가의 일일 수익률 계산
   - 첫 번째 행은 수익률이 NaN이므로 제거

4. **가격 Factor 계산**
   - `Close`가 0인 경우 1e-9로 대체 (0으로 나누기 방지)
   - `Open_factor = Open / Close`: 시가를 종가 대비 비율로 변환
   - `High_factor = High / Close`: 고가를 종가 대비 비율로 변환
   - `Low_factor = Low / Close`: 저가를 종가 대비 비율로 변환
   - 목적: 절대 가격 대신 상대적 가격 변동 패턴 추출

5. **데이터 정제**
   - 무한대 값(`np.inf`, `-np.inf`)을 NaN으로 변환
   - NaN 값이 포함된 행 제거 (`dropna()`)

**이미지 생성 로직 (`generate_image_from_window()`):**

1. **상대 가격 시리즈 생성**
   - 누적 수익률 계산: `RelClose = (1 + AdjReturn).cumprod()`
   - 첫날 종가를 1.0으로 정규화: `RelClose = RelClose / RelClose[0]`
   - 상대 가격 재구성:
     - `RelOpen = Open_factor × RelClose`
     - `RelHigh = High_factor × RelClose`
     - `RelLow = Low_factor × RelClose`

2. **이동평균 계산**
   - `MA = RelClose.rolling(window=n_days, min_periods=1).mean()`
   - n일 이미지에 대해 n일 이동평균선 계산

3. **스케일링 파라미터 계산**
   - 가격 범위: `min_price`, `max_price` (RelOpen, RelHigh, RelLow, RelClose, MA 중 최소/최대)
   - 거래량 범위: `max_volume` (해당 윈도우 내 최대 거래량)
   - 가격 범위가 0인 경우 1.0으로 설정 (0으로 나누기 방지)

4. **이미지 픽셀 매핑**
   - 이미지 크기: 높이 32px (가격 영역 25.6px + 거래량 영역 6.4px), 너비 15px (5일 × 3px/일)
   - 가격 스케일링: `y = (price_height - 1) × (1 - (price - min_price) / price_range)`
     - 상단이 높은 가격, 하단이 낮은 가격 (Y축 반전)
   - 거래량 스케일링: `height = (volume / max_volume) × (volume_height - 1)`

5. **차트 요소 그리기 (벡터화 최적화)**
   - **High-Low 바**: 각 날짜의 중앙 픽셀(x_center)에 y_high부터 y_low까지 수직선
   - **Open 점**: 각 날짜의 왼쪽 픽셀(x_left)에 y_open 위치에 점
   - **Close 점**: 각 날짜의 오른쪽 픽셀(x_right)에 y_close 위치에 점
   - **이동평균선**: OpenCV `cv2.line()`으로 연속된 점들을 선으로 연결
   - **거래량 바**: 하단 거래량 영역에 y_center 위치에서 아래로 거래량 높이만큼 수직선

**라벨 계산 로직 (`calculate_label_and_return()`):**

1. **누적 수익률 계산**
   - 향후 n일간의 수익률: `cum_ret_factor = (1 + AdjReturn).prod()`
   - 예: 5일간 각각 1%, 2%, -1%, 3%, 1% 수익 → `(1.01 × 1.02 × 0.99 × 1.03 × 1.01) ≈ 1.06`

2. **라벨 생성**
   - `cum_ret_factor > 1.0` → 라벨 = 1 (Up)
   - `cum_ret_factor ≤ 1.0` → 라벨 = 0 (Down)

3. **실제 수익률 계산**
   - `actual_return = cum_ret_factor - 1.0`
   - 예: `1.06 → 0.06` (6% 수익), `0.98 → -0.02` (-2% 손실)

**롤링 윈도우 처리 (`process_single_file()`):**

1. **윈도우 구성**
   - 이미지 윈도우: i일부터 (i + n_days - 1)일까지 (n_days일간)
   - 라벨 윈도우: (i + n_days)일부터 (i + 2×n_days - 1)일까지 (다음 n_days일간)
   - 최소 데이터 길이: `n_days + n_days = 2×n_days` (이미지 + 라벨)

2. **데이터 저장**
   - 날짜: 이미지 윈도우의 마지막 날짜 (예측 시점)
   - 형식: `'YYYY-MM-DD'` 문자열로 저장 (Timestamp 객체 대신)
   - 티커: 파일명에서 추출 (예: `AAPL.csv` → `AAPL`)

**최적화 기법:**

- **벡터화 연산**: NumPy 배열 연산으로 픽셀 그리기 최적화
- **멀티프로세싱**: `multiprocessing.Pool`로 여러 CSV 파일 병렬 처리
- **메모리 효율**: uint8 (이미지), float32 (수익률) 사용
- **에러 처리**: 빈 파일, 데이터 오류 등 예외 상황 처리

### `train.py`

CNN 모델을 구축하고 훈련합니다.

**모델 아키텍처:**

논문 기반의 2단계 Convolutional Neural Network 구조를 사용합니다.

```
입력 레이어
  Input Shape: (32, 15, 1)
  - 높이 32: 가격 영역(25.6px) + 거래량 영역(6.4px)
  - 너비 15: 5일간의 데이터 (하루당 3픽셀)
  - 채널 1: 흑백 이미지

Block 1: 첫 번째 Convolution Block
  Conv2D(filters=64, kernel_size=(5, 3), padding='same')
    → 출력: (32, 15, 64)
    → 파라미터 수: (5×3×1×64) + 64 = 1,024개
    → 5×3 커널로 시간적(너비) 및 가격적(높이) 패턴 추출
  
  BatchNormalization()
    → 출력: (32, 15, 64)
    → 내부 공변량 이동(Internal Covariate Shift) 감소
    → 훈련 안정성 향상
  
  LeakyReLU(alpha=0.01)
    → 출력: (32, 15, 64)
    → 음수 영역에서도 작은 기울기 유지 (Dying ReLU 문제 방지)
  
  MaxPooling2D(pool_size=(2, 1))
    → 출력: (16, 15, 64)
    → 높이만 절반으로 축소 (가격 차원 압축)
    → 너비는 유지 (시간 정보 보존)

Block 2: 두 번째 Convolution Block
  Conv2D(filters=128, kernel_size=(5, 3), padding='same')
    → 출력: (16, 15, 128)
    → 파라미터 수: (5×3×64×128) + 128 = 122,880개
    → 더 복잡한 고수준 패턴 추출
  
  BatchNormalization()
    → 출력: (16, 15, 128)
  
  LeakyReLU(alpha=0.01)
    → 출력: (16, 15, 128)
  
  MaxPooling2D(pool_size=(2, 1))
    → 출력: (8, 15, 128)
    → 최종 특징 맵 크기: 8 × 15 × 128 = 15,360차원

Fully Connected Head
  Flatten()
    → 출력: (15,360,)
    → 2D 특징 맵을 1D 벡터로 변환
  
  Dropout(rate=0.5)
    → 출력: (15,360,)
    → 훈련 시 50% 뉴런 무작위 비활성화
    → 과적합(Overfitting) 방지
  
  Dense(units=2, activation='softmax')
    → 출력: (2,)
    → 파라미터 수: (15,360 × 2) + 2 = 30,722개
    → 클래스별 확률 출력: [P(Down), P(Up)]
    → P(Down) + P(Up) = 1.0 (확률 정규화)
```

**레이어별 상세 정보:**

| 레이어 | 출력 크기 | 파라미터 수 | 설명 |
|--------|----------|------------|------|
| Input | (32, 15, 1) | 0 | 5일간의 OHLCV 차트 이미지 |
| Conv2D Block 1 | (32, 15, 64) | 1,024 | 저수준 패턴 (가격 변동, 캔들 형태) |
| MaxPool 1 | (16, 15, 64) | 0 | 공간 차원 축소 |
| Conv2D Block 2 | (16, 15, 128) | 122,880 | 고수준 패턴 (추세, 반전 신호) |
| MaxPool 2 | (8, 15, 128) | 0 | 최종 특징 압축 |
| Flatten | (15,360,) | 0 | 1D 벡터 변환 |
| Dropout | (15,360,) | 0 | 정규화 (50% 드롭) |
| Dense Output | (2,) | 30,722 | Up/Down 확률 |

**총 파라미터 수:** 약 154,626개

**훈련 설정:**
- Optimizer: Adam (learning_rate=1e-5)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Batch size: 128
- Epochs: 50 (Early Stopping 적용, patience=2)
- Validation split: 0.3 (훈련 데이터의 30%를 검증용으로 사용)
- 데이터 분할: 1993-2000 (훈련/검증), 2001- (테스트)

**정규화 기법:**
- Batch Normalization: 각 레이어의 입력 분포 정규화
- Dropout: Fully Connected 레이어에서 과적합 방지
- Early Stopping: 검증 손실이 개선되지 않으면 훈련 조기 종료

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
