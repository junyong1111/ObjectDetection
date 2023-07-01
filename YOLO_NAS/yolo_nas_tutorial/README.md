## YOLO_NAS Tutorial

---

## 1. 필요 라이브러리 다운로드

로컬에서 사용시 python 3.10 버전을 사용해야 한다 3.11 버전은 아직 에러가 많음

```bash
#-- requirements.txt
super-gradients==3.1.2
opencv-python
```

```bash
pip install -r requirements.txt
```

## 2. 코드 작성

**이미지에서 YOLO 사용**

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

**웹캠에서 YOLO 사용**

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

#-- 사전학습된 Yolo_nas_small 모델 불러오기(빠르지만 정확도가 낮음)
model = models.get("yolo_nas_s", pretrained_weights ="coco").to(device)
model.predict_webcam(conf =0.7)
```