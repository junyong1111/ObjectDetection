## SORT

SORT(Simple Online and Realtime Tracking) 알고리즘은 실시간 객체 추적(real-time object tracking)을 위해 개발된 알고리즘이다. SORT는 대규모 다중 객체 추적 문제를 해결하기 위한 효율적인 방법을 제공한다.

SORT 알고리즘은 먼저 객체 탐지(Detection) 단계를 통해 현재 프레임에서 객체를 감지한다. 객체 탐지 후, SORT 알고리즘은 추적(Tracking) 단계에서 이전 프레임에서 감지된 객체들과 현재 프레임에서 감지된 객체들을 매칭한다. 이를 위해 매칭 알고리즘인 헝가리안 알고리즘(Hungarian algorithm)을 사용한다. 헝가리안 알고리즘은 각 객체 간의 거리나 유사도를 기준으로 매칭을 수행하여 최적의 매칭을 찾아낸다.

SORT 알고리즘은 또한 객체의 속도와 크기를 추정하여 추적의 정확성을 향상시킨다. 객체의 속도와 크기 추정은 Kalman 필터(Kalman filter)와 함께 사용된다. Kalman 필터는 시스템의 상태를 추정하기 위한 재귀 필터링 기술로, 추적 중인 객체의 위치와 속도를 예측하고 업데이트하는 데 사용된다. 추적 단계에서 매칭된 객체들은 식별 번호(Track ID)를 할당받는다. 이를 통해 동일한 객체가 프레임 간에 일관되게 식별될 수 있다.

## Deep Sort

### Step0. 필요 라이브러리 다운로드

```bash
#-- requirements.txt
super-gradients==3.1.1
opencv-python
```

### Step1. deepsort 폴더 복사 후 같은 작업폴더에 붙여넣기

- https://github.com/AarohiSingla/DeepSORT-Object-Tracking

### Step2. yolo_nas_deepsort.py 파일 작성

- **필요 라이브러리 import**
    
    ```python
    #-- 필요 라이브러리 import
    import time
    import torch
    import cv2
    import torch.backends.cudnn as cudnn
    from PIL import Image
    import colorsys
    import numpy as np
    
    from super_gradients.training import models
    from super_gradients.common.object_names import Models
    
    from deep_sort.utils.parser import get_config
    from deep_sort.deep_sort import DeepSort
    from deep_sort.sort.tracker import Tracker
    ```
    
- 모델 설정 **및 GPU 설정**
    
    ```python
    #-- GPU 설정
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    #-- 모델 설정
    model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
    conf_treshold = 0.70
    ```
    
- **카메라 설정 및  deepsort설정**
    
    ```python
    #-- deep sort 알고리즘 설정
    deep_sort_weights = "deep_sort/deep/checkpoint/ckpt.t7"
    #-- max_age는 최대 몇 프레임까지 인정할지
    tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
    
    #-- video 설정
    video_path = "people.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error video file")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #-- 코덱 및 비디오 쓰기 설정
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    fuse_model=False
    
    frames = []
    i = 0
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()
    ```
    
- **반복문을 돌면서 동영상에서 people counting**
    
    ```python
    while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, (1200, 720))
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()
        
        with torch.no_grad():
            
            detection_pred = list(model.predict(frame, conf=conf_treshold)._images_prediction_lst)
            bboxes_xyxy = detection_pred[0].prediction.bboxes_xyxy.tolist()
            confidence = detection_pred[0].prediction.confidence.tolist()
            labels = [label for label in detection_pred[0].prediction.labels.tolist() if label == 0]
            class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
            labels = [int(label) for label in labels]
            class_name = list(set([class_names[index] for index in labels]))
            bboxes_xywh = []
            for bbox in bboxes_xyxy:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                bbox_xywh = [x1, y1, w, h]
                bboxes_xywh.append(bbox_xywh)
            bboxes_xywh = np.array(bboxes_xywh)
            tracks = tracker.update(bboxes_xywh, confidence, og_frame)
            
            for track in tracker.tracker.tracks:
                track_id = track.track_id
                hits = track.hits
                x1, y1, x2, y2 = track.to_tlbr()
                w = x2 - x1
                h = y2 - y1
                
                shift_percent = 0.50
                y_shift = int(h * shift_percent)
                x_shift = int(w * shift_percent)
                y1 += y_shift
                y2 += y_shift
                x1 += x_shift
                x2 += x_shift
                
                bbox_xywh = (x1, y1, w, h)
                color = (0, 255, 0)
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 +h)), color, 2)
                
                text_color = (0, 0, 0)
                cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
                
                text_color = (0, 0, 0)
            current_time = time.perf_counter()
            elapsed = (current_time - start_time)
            counter += 1
            if elapsed > 1:
                fps = counter / elapsed 
                counter = 0
                start_time = current_time
            cv2.putText(og_frame,
                        f"FPS: {np.round(fps, 2)}",
                        (10, int(h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            frames.append(og_frame)
            # out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
            out.write(og_frame)

            cv2.imshow("Video", og_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break
    ```
    
- **자원 반납**
    
    ```python
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    ```
    
- Output 비디오 확인
    
    ![Untitled](https://github.com/junyong1111/ObjectDetection/assets/79856225/ac4deeba-0673-4713-9678-beed67e3475f)
