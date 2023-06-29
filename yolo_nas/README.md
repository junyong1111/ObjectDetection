## YOLO-NAS

---

먼저 신경 구조 검색에 대해 살펴보겠습니다. **NAS(Nural Archtecture Search)**

신경 구조 검색은 세 가지 구성 요소로 이루어져 있다.

1. 첫 번째는 검색 공간 또는 검색 공간으로, 선택할 수 있는 유효한 아키텍처의 집합을 정의한다.
2. 두 번째 구성 요소는 검색 알고리즘으로, 검색 공간에서 가능한 아키텍처를 전송하는 방법을 담당하는 검색 알고리즘이다.
3. 세 번째 구성 요소는 평가 전략으로, 후보 아키텍처를 비교하는 데 사용되는 가치 평가 전략이다.

AutoNeck는 "자동 신경망 구성"을 의미하며, 객체 검출 모델을 탐색하기 위해 사용된다. 이를 위해 초기 검색 공간을 생성하고, NVIDIA T4에 최적화된 YOLO NAS의 최적 아키텍처를 찾기 위해 GPU를 3800시간 동안 사용했다. 또한, 저지연성이 필요한 실시간 객체 검출은 자율 주행차 등 다양한 응용 프로그램에서 중요하다. 그러나 에지 디바이스의 자원은 제한적이므로 클라우드가 아닌 디바이스에 모델을 배포하는 것은 어려움이 있다. 이런 제약사항을 극복하기 위해 "양자화" 기술이 사용된다. 양자화는 모델 가중치의 정밀도를 낮춰 메모리 사용량을 줄이고 실행 속도를 높이는 과정을 의미한다. YOLO NAS에서 사용된 양자화 기법은 int8 양자화로, 모델 가중치를 Float32에서 1바이트로 변환하여 메모리를 절약한다. 이를 위해 "EurVgg"라는 새로운 구성 요소를 사용하였으며, 이는 양자화 후 정확도 손실을 크게 개선하는 역할을 한다. 또한, quantization으로 인한 정확도 손실을 개선하기 위해 "Sharif Vgg" 블록이 사용되었으며, Uranus Hybrid Quantization 기술을 통해 모델의 특정 레이어에만 양자화를 적용하여 정보 손실과 지연 시간을 균형있게 조절하였다. 이러한 기술들을 통해 객체 검출 모델을 자동으로 탐색하고 최적화하여, 자동차, 휴대폰 등의 장치에 대규모 모델을 배포하면서도 저지연성을 유지할 수 있다.

![스크린샷 2023-06-28 오후 3.09.56.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07c68990-9bf6-4729-bc72-7a1c3d3488fc/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-06-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.09.56.png)

YOLO NAS는 기존 YOLO 모델의 한계인 양자화 지원 부족과 정확도 부족과 같은 중요한 요소들을 개선하기 위한 최신 딥 러닝 기술을 적용한 모델이다.

YOLO NAS의 핵심 구성 요소는 "백본(Backbone)", "넥(Neck)", "헤드(Head)"로 구성되어 있다. 백본은 입력 이미지에서 특징을 추출하는 부분으로, YOLO NAS는 밀집(dense) 블록과 희소(sparse) 블록을 결합한 이중 경로 백본을 사용한다. 넥은 백본에서 추출된 특징을 향상시키고 다양한 스케일에서 예측을 생성하는 부분이다. 다중 스케일 피라미드를 사용하여 백본의 다양한 레벨에서 특징을 결합하고 예측을 생성한다. 헤드는 모델의 최종 분류 및 회귀 작업을 수행하는 부분으로, 분류 분기와 회귀 분기로 구성되어 있다.

YOLO NAS 모델은 YOLO, YOLO NAS Small, YOLO NAS Medium, YOLO NAS Large와 같이 세 가지 다른 모델이 제공된다. 이 모델들은 유명한 Object365 벤치마크 데이터셋을 기반으로 사전 훈련되었으며, 추가적인 훈련 및 데이터 라벨링을 통해 성능을 향상시켰다. YOLO NAS 모델은 Roboflow A100 데이터셋에서 복잡한 객체 검출 작업을 처리하는 능력을 보였으며, 다른 YOLO 모델들보다 우수한 성능을 발휘한다고 소개되었다.

YOLO NAS는 Neural Architecture Search (NAS) 기술을 사용하여 개발된 모델로, 효율적이고 고성능의 딥 러닝 모델을 생성할 수 있다. Neural Architecture Search는 특정 작업에 대한 최적의 신경망 아키텍처를 자동으로 탐색하는 과정이다. 이를 위해 다양한 아키텍처의 탐색 공간을 탐색하고 가장 효율적이고 고성능인 아키텍처를 선택한다. YOLO NAS인 Uranus의 주요 특징을 살펴보면, 작은 객체의 검출 능력을 향상시키고 정확도를 향상시키며 계산 비율에 대한 성능을 높인다. 비교 결과로 YOLO v5, YOLO v7, YOLO v8 모델보다 Uranus가 우수한 성능을 보인다.

 YOLO NAS의 다른 주요 특징은 다음과 같다. 첫째로, YOLO NAS 아키텍처에는 양자화를 지원하는 블록과 최적화된 성능을 위한 선택적 양자화가 포함되어 있다. 둘째로, NAS 사전 훈련 모델은 Coco 데이터셋, Object365 데이터셋, Roboflow Hundred 데이터셋으로 사전 훈련되어 있으며, Roboflow Hundred 데이터셋에서 기존 YOLO 모델들보다 우수한 성능을 보인다. 마지막으로, 훈련 후 양자화 과정을 거쳐 YOLO NAS 모델을 Int8 양자화된 버전으로 변환하여 다른 YOLO 모델보다 더 효율적이다.

## 프로젝트 시작

---

## 1. 필요 라이브러리 다운로드

```bash
#-- requirements.txt
super-gradients==3.1.2
opencv-python
```

```bash
pip install -r requirements.txt
```

## 2. 코드 작성

```python
import cv2
import torch
from super_gradients.training import models

#-- GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
  print(torch.cuda.get_device_name(0))

#-- 이미지 불러오기
img = cv2.imread("이미지경로")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#-- 사전학습된 Yolo_nas_small 모델 불러오기(빠르지만 정확도가 낮음)
model_s = models.get("yolo_nas_s", pretrained_weights ="coco").to(device)
model_m = models.get("yolo_nas_m", pretrained_weights ="coco").to(device)
model_l = models.get("yolo_nas_l", pretrained_weights ="coco").to(device)

out_s = model_s.predict(img, conf = 0.3)
out_m = model_m.predict(img, conf = 0.3)
out_l = model_l.predict(img, conf = 0.3)

out_s.show()
out_m.show()
out_l.show()
```