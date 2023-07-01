## 프로젝트 시작(커스텀 데이터)

---

### 커스텀 데이터(Pokemon)

### Step0. 데이터 준비 및 라벨링을 위한 Roboflow 회원가입

- 포켓몬 데이터 셋(파이리, 꼬부기, 이상해씨)
    - 크롤링을 통해 이미지 데이터 확보
        
        [Images.zip](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b282b70c-8233-49b5-92bc-43041240abc3/Images.zip)
        
- Roboflow
    
    [Roboflow Universe: Open Source Computer Vision Community](https://universe.roboflow.com/)
    
    <img width="1676" alt="1" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/59275ae7-6cc0-485f-be1c-2bd21075f746">

### Step1. 데이터 라벨링

<img width="1594" alt="2" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/216591fb-8f0c-4d89-bd8f-2e094f5b9ad4">

<img width="683" alt="3" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/8bac1f92-84e7-474c-8ed7-c964bf6e0de7">

<img width="673" alt="4" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/3ea791cd-5ea9-4776-9cf0-0baf52c49a6d">

<img width="671" alt="5" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/7e81f4c0-064e-444f-a866-0a313bf8cf25">

<img width="1333" alt="6" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/3f468601-d055-4655-b12d-75f1bc8fbd48">

<img width="555" alt="7" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/e05baa5f-f732-4bb2-874e-7584c2cea4ac">

<img width="1060" alt="8" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/b84fba29-af6d-4ee0-b489-381d6cd9ac64">

<img width="1081" alt="9" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/2850f9a3-3875-4149-b0aa-f8cbaff44160">

<img width="1394" alt="10" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/837b2044-056f-4666-a626-c8e01cb6b63c">

**~~이미지 하나씩 라벨링 시작 smart 기능을 이용하면 더욱 정교하게 라벨링 가능(신세계 경험)~~**

<aside>
❗ 객체 탐지에서는 정교한 라벨링을 하면 인식을 못함 ㅠㅠ 사각형으로 다시 라벨링!!
Segmentation에서 했어야 하는듯…. 이 문제 때문에 몇 시간을 날림..

</aside>

- ~~만약 라벨링을 실수한경우 손바닥 모양을 클릭하면 쉽게 수정 가능!!~~

<img width="1622" alt="11" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/c8085ef1-03dc-45e7-9c2c-9657c231c7a3">


<img width="1345" alt="12" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/86347c28-7296-4d54-94c9-da4b15a4be7b">

**위처럼 하면 안됨 아래처럼 !! 사각형(바운딩 박스)으로 라벨링 해야함**

<img width="753" alt="13" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/2de95509-8348-4db5-8051-cff5f3663198">


**라벨링이 모두 끝나면 add 버튼 클릭**

<img width="1112" alt="14" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/326e4970-9b2b-40c9-abd6-7d370eba8275">


**원하는 비율로 데이터셋을 나누고 버튼 클릭(수정 가능)**

<img width="1108" alt="15" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/2124ac90-af78-42f4-9a30-f294bc7af619">


**데이터셋에서 이미지 증폭등 다양한 변환 가능**

<img width="1401" alt="16" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/8d9474ac-e70c-487b-b336-94b3a7d93f2c">

- 이미지 레벨과 바운딩박스 레벨에서 90도 변환 증폭을 적용해봤음
    
    <img width="829" alt="17" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/e385936c-20bc-426c-b833-ec5bee015afc">
    

**YOLOv5 format 으로 내보내기**

<img width="811" alt="18" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/4134acde-4200-4cab-acc6-0dacc7943889">

<img width="501" alt="19" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/73c3d4d3-11c5-47fc-aa38-6158361d4157">

**Colab 환경에서 학습을 해야 하므로 주피터 포맷을 선택하고 해당 코드 복사**

<img width="499" alt="20" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/9a54eee5-6089-4599-8b29-cead80558944">

### Step2. Colab에서 커스텀 데이터 학습

**Step0. 필요 라이브러리 설치**

- 아래 라이브러리 설치 후 런타임 재시작

```python
%%capture
#-- 현재 파이썬 3.10 버전부터는 생기는 에러가 있으므로 아래 3.1.1으로 다운받아야 에러가 안남
!pip install super-gradients==3.1.1
!pip install imutils
!pip install roboflow
!pip install pytube --upgrade
```

**Step1. 필요 라이브러리 import & GPU 설정**

```python
import cv2
import torch
from IPython.display import clear_output
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
#-- GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
  print(torch.cuda.get_device_name(0))
```

**Step2. 체크포인트 설정**

```python
CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='Pokémon_yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)
```

**Step3. roboflow 데이터를 코랩으로 가져오기**

- roboflow에서 복사해뒀던 코드를 붙여넣기

<img width="497" alt="21" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/fb86c031-ba41-443a-9611-456f67ebaf5a">


```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3NnsrtodD4CdBVrs1VJk")
project = rf.workspace("pokmon").project("pokemon-abicz")
dataset = project.version(1).download("yolov5")
```

위 코드를 실행하면 로컬에 해당 데이터셋 폴더가 생성 된다 해당 폴더 경로 복사

<img width="525" alt="22" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/d09c6922-c0a4-4cf8-a4b8-df8806498034">

**Step4. 디렉토리로 데이터셋 로드**

- data_dir은 위에서 복사한 경로를 넣어줌
- classes에는 클래스들을 넣어주면 된다.

```python
dataset_params = {
    'data_dir':'/content/pokemon-1',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['Bulbasaur', 'Charmander', 'Squirtle']
}
```

**Step5. 데이터 parmas를 데이터셋 parmas 인자로 삽입**

```python
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
        # 'show_all_warnings': True
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

clear_output()
```

**Step6. 앞에서 정의한 데이터 세트 검사**

```python
train_data.dataset.transforms
train_data.dataset.dataset_params['transforms'][1]
train_data.dataset.dataset_params['transforms'][1]['DetectionRandomAffine']['degrees'] = 10.42
```

**Step7. 증폭 기능이 적용된 훈련 데이터 시각화**

```python
train_data.dataset.plot()
```

![23](https://github.com/junyong1111/YOLO_NAS/assets/79856225/83168594-7fd4-424b-8607-c274eb480a33)

**Step8. 모델 인스턴스화**

- 다음은 미세 조정을 위해 모델을 인스턴스화하는 방법이다 여기에 `num_classes` 인수를 추가해야 한다는 점에 유의해야 한다.
- 이 튜토리얼에서는 `yolo_nas_s`를 사용하지만, 슈퍼 그레이디언트에는 두 가지 다른 종류의 yolo_nas_m`과`yolo_nas_l`를 사용할 수 있다.

```python
model = models.get('yolo_nas_s',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )
```

**Step9. 모델 하이퍼 파라미터 설정**

- `max_epochs` - 최대 훈련 에포크 수
- `loss` - 사용하려는 손실 함수
- `optimizer` - 사용하려는 손실 함수
- `train_metrics_list` - 트레이닝 중에 기록할 메트릭
- `valid_metrics_list` - 트레이닝 중에 기록할 메트릭
- `metric_to_watch` - 모델 체크포인트가 저장될 지표

다음과 같은 다양한 `옵티마이저` 중에서 선택할 수 있다: Adam, AdamW, SGD, Lion 또는 RMSProps. 이러한 옵티마이저의 잘못된 파라미터를 변경하려면 해당 파라미터를 `optimizer_params`에 전달한다.

```python
train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # 이 예제의 경우 15개의 에포크만 교육한다.
    "max_epochs": 15,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # 참고: num_classes는 여기에 정의되어야 한다.
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # 참고: num_classes는 여기에 정의되어야 한다.
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}
```

**Step10. 모델 학습**

```python
trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)
```

**Step11. 최고의 모델 얻기**

```python
best_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/Pokemon_yolonas_run/ckpt_best.pth")
```

**Step12. 모델 평가**

```python
trainer.test(model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                   top_k_predictions=300,
                                                   num_cls=len(dataset_params['classes']),
                                                   normalize_targets=True,
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                          nms_top_k=1000,
                                                                                                          max_predictions=300,
                                                                                                          nms_threshold=0.7)
                                                  ))
```

**Step13. 모델 예측**

```python
#-- 테스트 동영상 다운로드
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!gdown "https://drive.google.com/uc?id=1HsCBy8HU0Rqs-nb_mScRXVmLwrY1G3UQ"
```

```python
input_video_path = f"/content/pokemon.mp4"
output_video_path = "detections.mp4"

best_model.to(device)
best_model.predict(input_video_path, conf = 0.4).save(output_video_path)
```

```python
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/detections.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

<img width="605" alt="24" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/73d61c11-62d5-4da4-b15f-20957169f989">

small 모델과 적은 학습으로 인해 높은 결과는 나오지는 않는다. 파라메터 수정 후 테스트 해보면 좋을듯

### Large + 100번 반복 후 84퍼 정확도 모델 결과

확실하게 이전 모델보다는 잘 인식한다. 더 많은 데이터셋과 파라메터 수정을 하면 좋은 결과 가능할듯

<img width="626" alt="25" src="https://github.com/junyong1111/YOLO_NAS/assets/79856225/0a1db4a9-ed42-4ad8-b7e5-98fb1b6027bf">
