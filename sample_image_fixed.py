import numpy as np
import matplotlib.pyplot as plt

# 1. 불러올 NPZ 파일 이름 (수정된 파일)
# (이 파일이 create_dataset_fixed.py와 동일한 경로에 있다고 가정)
NPZ_FILE = 'data_L5_R5_with_returns_FIXED.npz' 

# 2. 확인할 랜덤 샘플 개수
NUM_SAMPLES = 5

data = None # data 객체를 try/finally에서 모두 접근할 수 있도록 초기화

try:
    print(f"'{NPZ_FILE}' 파일 로드 중 (mmap_mode='r')...")
    # mmap_mode='r' : 파일을 메모리에 올리지 않고, 디스크에 연결만 합니다.
    data = np.load(NPZ_FILE, allow_pickle=True, mmap_mode='r')
    
    # 3. 데이터 배열 '포인터' 가져오기 (이 시점엔 메모리 차지 안 함)
    images = data['images']
    labels = data['labels']
    dates = data['dates']
    returns = data['returns']
    
    total_count = len(images)
    
    if total_count == 0:
        print("오류: 파일에 데이터가 없습니다.")
    else:
        print(f"파일 로드 성공. 총 {total_count}개의 샘플 발견.")

        # 4. 전체 샘플 중 NUM_SAMPLES 개수만큼 랜덤 인덱스 추출
        # replace=False : 중복 없이 뽑기
        random_indices = np.random.choice(total_count, NUM_SAMPLES, replace=False)
        random_indices.sort() # 보기 좋게 정렬
        
        print(f"\n--- {NUM_SAMPLES}개의 랜덤 샘플 정보 (인덱스: {random_indices}) ---")

        # 5. 랜덤 인덱스를 하나씩 돌면서 "실제로" 데이터 읽기
        for i, index in enumerate(random_indices):
            print(f"\n--- {i+1}번째 샘플 (전체 인덱스: {index}) ---")
            
            # 🚨 이 시점에 디스크에서 딱 해당 인덱스의 데이터만 읽어옵니다.
            sample_image = images[index]
            sample_label = labels[index]
            sample_date = dates[index]
            sample_return = returns[index]
            
            print(f"  - 날짜 (Date): {sample_date}")
            print(f"  - 라벨 (Label): {sample_label} (0=Down, 1=Up)")
            print(f"  - 실제 수익률 (Return): {sample_return:.4f}")
            print(f"  - 이미지 형태: {sample_image.shape}")
            
            # 6. 이미지 시각화
            plt.figure(figsize=(6, 4))
            # (32, 15, 1) 형태를 (32, 15)로 변경하여 흑백 이미지로 표시
            plt.imshow(np.squeeze(sample_image), cmap='gray', aspect='auto')
            plt.title(f"Sample Index: {index} | Date: {sample_date} | Label: {sample_label}")
            plt.xlabel("Features (Time steps)")
            plt.ylabel("Channels (LOB data)")
            plt.show()

except FileNotFoundError:
    print(f"오류: '{NPZ_FILE}' 파일을 찾을 수 없습니다.")
except KeyError:
    print("오류: .npz 파일에 'images', 'labels', 'dates', 'returns' 키 중 하나가 없습니다.")
except Exception as e:
    print(f"파일 로드 중 알 수 없는 오류 발생: {e}")

finally:
    # 7. (매우 중요) mmap_mode로 열었으면 반드시 닫아주어야 합니다.
    if data is not None and hasattr(data, 'close'):
        data.close()
        print("\n파일 핸들(mmap)을 닫았습니다.")