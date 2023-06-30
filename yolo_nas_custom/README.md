## 프로젝트 시작(커스텀 데이터)

---

### 커스텀 데이터(Pokemon)

### Step0. 데이터 준비 및 라벨링을 위한 Roboflow 회원가입

- 포켓몬 데이터 셋(파이리, 꼬부기, 이상해씨)
    - 크롤링을 통해 이미지 데이터 확보
        
        [Images.zip](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b282b70c-8233-49b5-92bc-43041240abc3/Images.zip)
        
- Roboflow
    
    [Roboflow Universe: Open Source Computer Vision Community](https://universe.roboflow.com/)
    
    <img width="1676" alt="1" src="https://github.com/junyong1111/objectdetection/assets/79856225/48464af7-cf7d-4091-9e10-3fcb85e4ebf3">

    

### Step1. 데이터 라벨링

<img width="1594" alt="2" src="https://github.com/junyong1111/objectdetection/assets/79856225/47f2df49-e787-4e01-bbc3-17c826cd623a">

<img width="683" alt="3" src="https://github.com/junyong1111/objectdetection/assets/79856225/542a4cf1-ff1a-4464-9852-b7b8cb7c4f2b">

<img width="673" alt="4" src="https://github.com/junyong1111/objectdetection/assets/79856225/af1d00df-a78c-432f-8ceb-a9ed27d3ad4f">

<img width="671" alt="5" src="https://github.com/junyong1111/objectdetection/assets/79856225/73169c00-33b4-4db2-906f-bd4fd6a57c6e">

<img width="1333" alt="6" src="https://github.com/junyong1111/objectdetection/assets/79856225/22adc5af-7ee1-487c-9c90-32ea4da1b340">

<img width="555" alt="7" src="https://github.com/junyong1111/objectdetection/assets/79856225/9af634c8-696e-4ba3-9c4a-fa6c0b385576">

<img width="1060" alt="8" src="https://github.com/junyong1111/objectdetection/assets/79856225/bdedaf17-25bc-48fd-93f1-c1abc5f8526a">


<img width="1081" alt="9" src="https://github.com/junyong1111/objectdetection/assets/79856225/eea73226-dfa4-46b8-b59b-51ae3ecf36fb">


<img width="1394" alt="10" src="https://github.com/junyong1111/objectdetection/assets/79856225/74123734-7d12-4f48-8e2e-94ea379758a9">

**~~이미지 하나씩 라벨링 시작 smart 기능을 이용하면 더욱 정교하게 라벨링 가능(신세계 경험)~~**

<aside>
❗ **객체 탐지에서는 정교한 라벨링을 하면 인식을 못함 ㅠㅠ 사각형으로 다시 라벨링!!
Segmentation에서 했어야 하는듯…. 이 문제 때문에 몇 시간을 날림..**

</aside>

- ~~만약 라벨링을 실수한경우 손바닥 모양을 클릭하면 쉽게 수정 가능!!~~

<img width="1622" alt="11" src="https://github.com/junyong1111/objectdetection/assets/79856225/829ac7f2-4457-49ac-bc66-28bd100d5fc5">

<img width="1345" alt="12" src="https://github.com/junyong1111/objectdetection/assets/79856225/c29bb269-ec7e-4c7e-8117-5dfdb5928b23">

**위처럼 하면 안됨 아래처럼 !! 사각형(바운딩 박스)으로 라벨링 해야함**

<img width="753" alt="13" src="https://github.com/junyong1111/objectdetection/assets/79856225/5c714853-54a9-4327-b7ed-231f3daa5e0f">

**라벨링이 모두 끝나면 add 버튼 클릭**

<img width="1112" alt="14" src="https://github.com/junyong1111/objectdetection/assets/79856225/8449aee9-f841-4c04-9ab5-36d6350bf16b">

**원하는 비율로 데이터셋을 나누고 버튼 클릭(수정 가능)**

<img width="1108" alt="15" src="https://github.com/junyong1111/objectdetection/assets/79856225/4ee3ce23-b8b6-4847-83b0-15e927ad2840">


**데이터셋에서 이미지 증폭등 다양한 변환 가능**

<img width="1401" alt="16" src="https://github.com/junyong1111/objectdetection/assets/79856225/5b9017ff-efb7-40fb-b8ea-6ce960d05706">

- 이미지 레벨과 바운딩박스 레벨에서 90도 변환 증폭을 적용해봤음
    
    <img width="829" alt="17" src="https://github.com/junyong1111/objectdetection/assets/79856225/fd47e2d6-58de-4cc1-9536-f8dfd9deb9e3">
    

**YOLOv5 format 으로 내보내기**

<img width="811" alt="18" src="https://github.com/junyong1111/objectdetection/assets/79856225/af4d6a8d-8095-4f45-89ea-d23212335e2e">

<img width="501" alt="19" src="https://github.com/junyong1111/objectdetection/assets/79856225/7c44e1ae-ae5a-4ff0-aaf6-fe4bf6ee3ea4">

**Colab 환경에서 학습을 해야 하므로 주피터 포맷을 선택하고 해당 코드 복사**

<img width="499" alt="20" src="https://github.com/junyong1111/objectdetection/assets/79856225/cb0650c1-bd63-4f09-be19-2c428d74a35d">


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



```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3NnsrtodD4CdBVrs1VJk")
project = rf.workspace("pokmon").project("pokemon-abicz")
dataset = project.version(1).download("yolov5")
```

위 코드를 실행하면 로컬에 해당 데이터셋 폴더가 생성 된다 해당 폴더 경로 복사

<img width="525" alt="21" src="https://github.com/junyong1111/objectdetection/assets/79856225/338fdb64-83b8-4686-b4db-94c1c8524e33">




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

![22](https://github.com/junyong1111/objectdetection/assets/79856225/f26e88d7-8770-4008-a220-221e8d983c0d)


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
!gdown "https://drive.google.com/uc?id=1HsCBy8HU0Rqs-nb_mScRXVmLwrY1G3UQ"
```

```python
input_video_path = f"/content/pokemon.mp4"
output_video_path = "detections.mp4"

best_model.to(device).predict(input_video_path).save(output_video_path)
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

<img width="605" alt="23" src="https://github.com/junyong1111/objectdetection/assets/79856225/5e53c265-6025-4d92-bb8b-dbcd8ccb997c">
![22](https://github.com/junyong1111/objectdetection/assets/79856225/039f9c5a-dcce-4693-864f-8133bf1ab9ef)

small 모델과 적은 학습으로 인해 높은 결과는 나오지는 않는다. 파라메터 수정 후 테스트 해보면 좋을듯