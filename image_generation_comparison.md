# 이미지 생성 로직 비교 분석

## 1. 가격 정규화 방식 (가장 큰 차이점)

### Colab 코드 (`make_stock_image`)
```python
# 원본 가격 데이터를 직접 사용
price_cols = ['Open', 'High', 'Low', 'Close', 'MA']
price_min = df[price_cols].min().min()
price_max = df[price_cols].max().max()
price_range = price_max - price_min
price_slice = (df[price_cols] - price_min) / price_range
```
- **방식**: 윈도우 내 최소/최대값으로 정규화 (Min-Max Scaling)
- **특징**: 절대 가격 기준, 윈도우마다 스케일이 달라짐

### make_npz.py (`generate_image_from_window`)
```python
# 상대 가격 시리즈 생성 (누적 수익률 기반)
rel_prices['RelClose'] = (1 + rel_prices['AdjReturn']).cumprod()
first_rel_close = rel_prices['RelClose'].iloc[0]
rel_prices['RelClose'] = rel_prices['RelClose'] / first_rel_close

# O, H, L을 상대 가격으로 재구성
rel_prices['RelOpen'] = rel_prices['Open_factor'] * rel_prices['RelClose']
rel_prices['RelHigh'] = rel_prices['High_factor'] * rel_prices['RelClose']
rel_prices['RelLow'] = rel_prices['Low_factor'] * rel_prices['RelClose']
```
- **방식**: 첫날 종가를 1로 정규화한 상대 가격 (Relative Price)
- **특징**: 수익률 기반, 첫날 대비 상대적 변화를 표현

---

## 2. 이동평균(MA) 계산

### Colab 코드
```python
df["MA"] = df["Close"].rolling(window = ma_period, min_periods = 1).mean()
```
- **기준**: 원본 `Close` 가격으로 계산
- **기간**: `ma_period=20` (고정값)

### make_npz.py
```python
rel_prices[f'MA'] = rel_prices['RelClose'].rolling(window=n_days, min_periods=1).mean()
```
- **기준**: 상대 종가(`RelClose`)로 계산
- **기간**: `n_days` (이미지 윈도우 크기와 동일)

---

## 3. 이미지 그리기 방식 (픽셀 배치)

### Colab 코드
```python
for i in range(win_size):
    # 시가: i*3 위치
    image[price_slice.loc[i, 'Open'], i * 3] = 255
    
    # 고가-저가: i*3+1 위치
    image[low:high+1, i * 3 + 1] = 255
    
    # MA: i*3+2 위치 (회색 128)
    image[price_slice.loc[i, 'MA'], i * 3 + 2] = 128
    
    # 종가: i*3+2 위치 (흰색 255, MA 위에 덮어씀)
    image[price_slice.loc[i, 'Close'], i * 3 + 2] = 255
```
- **배치**: Open(왼쪽), High-Low(중앙), MA+Close(오른쪽, 겹침)
- **MA 표시**: 회색(128)으로 먼저 그린 후 종가(255)로 덮어씀

### make_npz.py
```python
x_left_arr = np.arange(n_days) * 3      # Open 위치
x_center_arr = x_left_arr + 1           # High-Low 위치
x_right_arr = x_left_arr + 2            # Close 위치

# Open: x_left
image[y_open_arr[t], x_left] = 255

# High-Low: x_center
image[y_high:y_low+1, x_center] = 255

# Close: x_right
image[y_close_arr[t], x_right] = 255

# MA: cv2.polylines로 연결된 선
cv2.polylines(image, [pts], isClosed=False, color=255, thickness=1)
```
- **배치**: Open(왼쪽), High-Low(중앙), Close(오른쪽), MA(연결선)
- **MA 표시**: `cv2.polylines`로 점들을 연결한 연속선

---

## 4. 거래량 처리

### Colab 코드
```python
volume_slice = (df[['Volume']] - volume_min) / volume_range
volume_slice = volume_slice.apply(lambda x: (vol_height - 1) - (x * (vol_height - 1)).astype(int))

# 거래량 바 그리기
vol = volume_slice.loc[i, 'Volume']
image[vol_base + vol:, i * 3 + 1] = 255
```
- **정규화**: Min-Max Scaling
- **위치**: `vol_base + vol`부터 아래로 그리기
- **영역**: `vol_base=21` (32px 이미지 기준)

### make_npz.py
```python
def scale_volume_h_vec(volumes):
    if max_volume == 0:
        return np.zeros_like(volumes, dtype=np.int32)
    return ((volumes / max_volume) * (volume_height - 1)).astype(np.int32)

# 거래량 바 그리기
vol_h = vol_h_arr[t]
if vol_h > 0:
    image[total_height - vol_h : total_height, x_left : x_right + 1] = 255
```
- **정규화**: 최대값 대비 비율
- **위치**: 이미지 하단(`total_height`)에서 위로 그리기
- **영역**: 전체 너비(`x_left : x_right + 1`) 사용

---

## 5. 이미지 크기 및 영역 분할

### Colab 코드
```python
if image_size[0] == 32:
    price_height, vol_base, vol_height = 20, 21, 7
else:
    price_height, vol_base, vol_height = 40, 41, 13
```
- **고정값**: 이미지 크기에 따라 하드코딩된 값 사용

### make_npz.py
```python
price_height = int(total_height * 4 / 5)  # 가격 영역: 4/5
volume_height = total_height - price_height  # 거래량 영역: 1/5
```
- **비율 기반**: 전체 높이의 4:1 비율로 동적 계산

---

## 6. 데이터 전처리

### Colab 코드
```python
def preprocess(df, ma_period=20):
    df["Date"] = pd.to_datetime(df["Date"])
    df["MA"] = df["Close"].rolling(window = ma_period, min_periods = 1).mean()
    df["ret1"] = df["Close"].pct_change(1).shift(-1) * 100
    # ...
```
- **수익률**: 미래 수익률(`shift(-1)`) 계산
- **가격**: 원본 가격 그대로 사용

### make_npz.py
```python
def preprocess_data(df):
    df['AdjReturn'] = df['Adj Close'].pct_change()
    df['Open_factor'] = df['Open'] / df['Close']
    df['High_factor'] = df['High'] / df['Close']
    df['Low_factor'] = df['Low'] / df['Close']
    # ...
```
- **수익률**: 과거 수익률(`pct_change()`) 계산
- **가격**: 종가 대비 비율(factor)로 변환 후 상대 가격 재구성

---

## 요약: 핵심 차이점

| 항목 | Colab 코드 | make_npz.py |
|------|-----------|-------------|
| **가격 정규화** | Min-Max (절대 가격) | 상대 가격 (첫날=1 기준) |
| **MA 계산 기준** | 원본 Close | 상대 Close (RelClose) |
| **MA 그리기** | 점(128) + 종가 덮어쓰기 | 연결선 (cv2.polylines) |
| **거래량 위치** | vol_base + vol부터 아래 | 하단에서 위로 |
| **이미지 영역** | 고정값 (20/7 또는 40/13) | 비율 기반 (4:1) |
| **데이터 소스** | 원본 가격 | 조정 종가 기반 상대 가격 |

---

## 영향 분석

1. **상대 가격 vs 절대 가격**: 
   - make_npz.py는 가격의 절대값보다 상대적 변화에 집중
   - Colab 코드는 윈도우 내 가격 범위에 따라 스케일이 달라짐

2. **MA 표현 방식**:
   - make_npz.py의 연결선 방식이 더 시각적으로 명확
   - Colab 코드는 MA와 종가가 같은 위치에 겹쳐서 표시

3. **일관성**:
   - make_npz.py는 논문 기반의 일관된 스케일링 방식 사용
   - Colab 코드는 윈도우별로 다른 스케일 적용 가능

