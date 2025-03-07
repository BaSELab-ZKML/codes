# COVID-19 X-ray 분류 모델 설명

## 1. 데이터 준비
- 이 코드는 COVID-19 X-ray 이미지를 학습하여 감염 여부를 분류하는 모델을 만드는 과정입니다.
- 이미지는 픽셀 값이 0~1 사이가 되도록 정규화(`rescale=1./255`)하여 모델이 더 쉽게 학습할 수 있도록 준비합니다.
- 데이터를 학습용(80%)과 검증용(20%)으로 나누어, **검증용 데이터셋을 통해 학습 중 모델이 얼마나 잘 작동하는지 확인**할 수 있습니다.

## 2. 데이터 불러오기
- 학습 데이터(`train_generator`)와 검증 데이터(`val_generator`)를 불러옵니다.
- 이미지는 `(224, 224)` 크기로 변환됩니다. 이는 모델이 처리할 수 있는 고정된 크기를 맞추기 위함입니다.
- 데이터는 16개씩 묶어서 처리되며(배치 크기 16), 학습 중 데이터를 무작위로 섞어(`shuffle=True`) 모델이 특정 패턴에 치우치지 않도록 합니다.
- 이 데이터는 총 4개의 클래스(카테고리)로 나뉘어 있습니다. (정상, 코로나, 폐렴, 폐결핵)

## 3. 모델 구성

### 첫 번째 Conv2D 층 (`Conv2D(16, (3,3))`)
- 입력 데이터는 `(224, 224, 3)` 크기(이미지 크기와 RGB 채널)입니다.
- 첫 번째 합성곱 층은 16개의 필터를 사용하여 이미지의 기본적인 특징(예: 엣지, 선 등)을 추출합니다.
- 필터 크기 `(3, 3)`은 이미지의 작은 부분(3x3 영역)을 분석하여 유용한 정보를 학습합니다.
- 활성화 함수 `relu`는 음수 값을 제거하여 데이터를 더 효과적으로 학습할 수 있도록 합니다.

### 첫 번째 MaxPooling2D 층 (`MaxPooling2D((2,2))`)
- 데이터 크기를 절반으로 줄여 `(112, 112, 16)`으로 만듭니다.
- 여기서 `(112, 112)`는 이미지의 가로와 세로 크기이고, `16`은 필터의 개수를 나타냅니다.
- 이렇게 하면 가장 중요한 특징만 남기고 불필요한 세부 정보를 제거합니다.

### 두 번째 Conv2D 층 (`Conv2D(32, (3,3))`)
- 이 층은 32개의 필터를 사용하여 더 복잡한 패턴과 윤곽선을 학습합니다.
- 출력 크기는 `(112, 112, 32)`이며, `(112, 112)`는 이미지 크기, `32`는 필터의 개수입니다.

### 두 번째 MaxPooling2D 층
- 데이터를 다시 절반으로 줄여 `(56, 56, 32)`로 만듭니다.
- 여기서 `(56, 56)`은 이미지 크기, `32`는 필터의 개수를 나타냅니다.

### 세 번째 Conv2D 층 (`Conv2D(64, (3,3))`)
- 64개의 필터를 사용하여 텍스처, 작은 디테일 같은 더 복잡한 특징을 학습합니다.
- 출력 크기는 `(56, 56, 64)`이며, `(56, 56)`은 이미지 크기, `64`는 필터의 개수입니다.

### 세 번째 MaxPooling2D 층
- 이미지 크기를 다시 절반으로 줄여 `(28, 28, 64)`로 만듭니다.
- 여기서 `(28, 28)`은 이미지 크기, `64`는 필터의 개수를 나타냅니다.

### Flatten 층
- 3차원 데이터를 1차원으로 펼쳐 `(50176,)` 크기로 만듭니다. 이는 이후 단계에서 데이터를 처리하기 쉽게 만들기 위함입니다.

### 첫 번째 Dense 층 (`Dense(64, activation='relu')`)
- 64개의 뉴런으로 구성된 이 층은 데이터를 분석하고, 가장 중요한 특징을 학습합니다.
- 이 층은 데이터의 복잡한 관계를 이해하는 데 중요한 역할을 합니다.

### 출력층 (`Dense(4, activation='softmax')`)
- 4개의 뉴런을 사용해 최종 결과를 출력합니다. 각 뉴런은 4개의 클래스 중 하나를 나타냅니다.
- `softmax` 함수는 각 클래스에 대한 확률을 계산하여 가장 가능성이 높은 클래스를 예측합니다.

## 4. 필터 수 증가: 16 → 32 → 64
- **필터 수를 점진적으로 늘리는 이유**
  - 처음엔 간단한 특징(예: 엣지, 선)을 학습하고, 점점 더 복잡한 특징(예: 패턴, 세부 구조)을 학습합니다.
  - 필터 수가 많아질수록 데이터에서 더 많은 정보를 추출할 수 있습니다.
- **왜 2의 배수를 사용하는가?**
  - 2의 배수는 GPU 메모리 관리와 병렬 계산에 최적화되어 학습 속도를 높이는 데 유리합니다.

## 5. 모델 컴파일
- **손실 함수(`categorical_crossentropy`)**
  - 모델이 예측한 값과 실제 값의 차이를 계산하여 학습에 반영합니다. 다중 클래스 분류 문제에 적합합니다.
- **옵티마이저(`Adam`)**
  - 학습 속도를 조정하며 효율적으로 가중치를 업데이트합니다. 초기 학습률은 0.001로 설정되었습니다.
- **평가지표(`accuracy`)**
  - 모델의 정확도를 평가하여 학습이 잘 진행되고 있는지 확인합니다.

## 6. 콜백 설정
- **ModelCheckpoint**
  - 학습 중 검증 데이터 손실(`val_loss`)이 가장 낮았던 시점의 모델을 저장합니다.
  - 저장된 파일은 `best_small_model.h5`로 이름이 지정됩니다.
- **EarlyStopping**
  - 학습 도중 성능이 더 이상 개선되지 않으면(예: 5번의 반복 동안 개선 없음) 학습을 중단합니다.
  - 이렇게 하면 불필요한 반복을 줄이고, 최적의 가중치를 복원합니다.

## 7. 모델 학습 및 저장
- 모델은 최대 50번 반복(에포크)하여 학습합니다.
- 각 에포크에서는 학습 데이터로 학습하고, 검증 데이터를 사용해 성능을 평가합니다.
- 학습이 끝난 후, 최종 모델은 `covid_classification_final_small.h5`로 저장됩니다.
- 이 저장된 모델은 나중에 새로운 데이터를 예측하는 데 사용됩니다.


