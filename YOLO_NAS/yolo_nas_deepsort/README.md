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
    while True:
        ret, frame = cap.read()  # 비디오 프레임 읽기
    
        count += 1  # 프레임 카운트 증가
    
        if ret:
            detections = np.empty((0, 5))
    
            # 모델을 사용하여 객체 검출 및 추적 수행
            result = list(model.predict(frame, conf=0.35))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()  # 객체의 경계상자 좌표
            confidences = result.prediction.confidence  # 객체의 신뢰도
            labels = result.prediction.labels.tolist()  # 객체의 레이블
    
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence*100))/100
    
                if class_name == "person" and conf > 0.3:
                    currentarray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentarray))
    
            resultsTracker = tracker.update(detections)  # 객체 추적 업데이트
    
            # 경계선 그리기
            cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (255,0,0), 3)  # 상한선
            cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (255,0,0), 3)  # 하한선
    
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
                # 객체를 사각형으로 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 3)
    
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    
                label = f'{int(id)}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
    
                # 객체 ID와 함께 사각형 위에 텍스트 표시
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
                # 상한선과 하한선을 통과한 객체 수 계산 및 표시
                if limitup[0] < cx < limitup[2] and limitup[1] - 15 < cy < limitup[3] + 15:
                    if totalCountUp.count(id) == 0:
                        totalCountUp.append(id)
                        cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (0, 255, 0), 3)
    
                if limitdown[0] < cx < limitdown[2] and limitdown[1] - 15 < cy < limitdown[3] + 15:
                    if totalCountDown.count(id) == 0:
                        totalCountDown.append(id)
                        cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (0, 255, 0), 3)
    
            # 상단 영역에 인원 수 표시
            cv2.rectangle(frame, (100, 65), (441, 97), [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, str("Person Entering") + ":" + str(len(totalCountUp)), (141, 91), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
            # 하단 영역에 인원 수 표시
            cv2.rectangle(frame, (710, 65), (1100, 97), [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, str("Person Leaving") + ":" + str(len(totalCountDown)), (741, 91), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
            resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            out.write(frame)
    
            cv2.imshow("Frame", frame)
    
            if cv2.waitKey(1) & 0xFF == ord('1'):  # '1' 키를 누르면 반복문 종료
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