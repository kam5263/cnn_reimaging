import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 확인할 .npz 파일 이름
npz_file = 'data_L5_R5_nyse_test.npz'

# 2. 확인할 샘플 번호
sample_index = 200 

# 3. 저장할 이미지 파일 이름
output_image_file = 'check_L5_R5_sample3.png'

print(f"'{npz_file}' 파일 로드 중...")

# 4. 파일 로드
try:
    data = np.load(npz_file, allow_pickle=True)
except FileNotFoundError:
    print(f"오류: '{npz_file}' 파일을 찾을 수 없습니다.")
    exit()

# 5. npz 파일 내부의 데이터 배열에 접근
# (저장 시 사용한 'images', 'labels', 'dates', 'returns', 'tickers' 키를 사용합니다)
try:
    images = data['images']
    labels = data['labels']
    dates = data['dates']
    returns = data['returns']
    tickers = data['tickers']
except KeyError:
    print("오류: .npz 파일에 'images', 'labels', 'dates', 'returns', 'tickers' 키가 없습니다.")
    data.close()
    exit()

print(f"파일 로드 성공. 총 {len(images)}개의 샘플이 있습니다.")
print(f"  이미지(X) 형태: {images.shape}") # (N, 32, 15, 1)

# 6. 샘플 데이터 확인
if len(images) > sample_index:
    
    # 샘플 이미지, 라벨, 날짜, 수익률, 티커 가져오기
    sample_image = images[sample_index]
    sample_label = labels[sample_index]
    sample_date = dates[sample_index]
    sample_return = returns[sample_index]
    sample_ticker = tickers[sample_index]

    # 날짜 포맷팅 (타입에 따라 다르게 처리)
    if isinstance(sample_date, np.datetime64):
        date_str = np.datetime_as_string(sample_date, unit='D')
    elif isinstance(sample_date, str):
        date_str = sample_date
    else:
        date_str = str(sample_date)
    
    # 티커 포맷팅 (문자열로 변환)
    if isinstance(sample_ticker, (bytes, np.bytes_)):
        ticker_str = sample_ticker.decode('utf-8')
    elif isinstance(sample_ticker, np.str_):
        ticker_str = str(sample_ticker)
    else:
        ticker_str = str(sample_ticker)

    print(f"\n--- {sample_index}번째 샘플 정보 ---")
    print(f"  티커: {ticker_str}")
    print(f"  날짜: {date_str}") # 날짜 포맷팅
    print(f"  라벨: {sample_label} ({'UP' if sample_label == 1 else 'DOWN/STAY'})")
    print(f"  실제 수익률: {sample_return:.4f} ({sample_return*100:.2f}%)")

    # 7. 이미지 저장 (matplotlib 사용)
    
    # (32, 15, 1) 형태의 이미지를 (32, 15)로 변경 (squeeze)
    img_to_save = sample_image.squeeze()

    # 5일 이미지는 32x15 픽셀로 매우 작습니다.
    # 그대로 저장하면 너무 작아서 보이지 않으므로, figsize와 dpi를 조절하여
    # 사람이 볼 수 있도록 크게 저장합니다.
    # (너비:높이 비율 = 15:32)
    plt.figure(figsize=(3, 6.4)) # 너비 3인치, 높이 6.4인치
    
    # interpolation='nearest'는 픽셀을 흐릿하게 만들지 않고 확대합니다.
    plt.imshow(img_to_save, cmap='gray', interpolation='nearest')
    
    # 이미지에 제목으로 티커, 라벨, 날짜, 수익률 추가
    plt.title(f"Ticker: {ticker_str}\nSample: {sample_index} | Date: {date_str}\nLabel: {sample_label} | Return: {sample_return:.4f} ({sample_return*100:.2f}%)")
    
    # 파일로 저장
    plt.savefig(output_image_file, dpi=100) # dpi=100 -> (300x640) 픽셀 크기
    
    print(f"\n샘플 이미지를 '{output_image_file}' 파일로 저장했습니다.")
    
else:
    print(f"\n오류: 이미지가 {sample_index}개 미만입니다.")

# 8. 파일 핸들러 닫기
data.close()