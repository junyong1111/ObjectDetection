import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

#-- 카메라 설정
cap = cv2.VideoCapture("people.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

#-- GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#-- 모델 설정
model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

count = 0
classNames = ["person"]


while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100
            label = f'{class_name}{conf}'
            print("Frame N", count, "", x1, y1,x2, y2)
            t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] -3
            cv2.rectangle(frame, (x1, y1), c2, [255,144, 30], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
						#-- 인식된 객체 블러 처리 하는 코드
            frame_area = frame[y1:y2, x1:x2]
            blur = cv2.blur(frame_area, (20,20))
            frame[int(y1):int(y2), int(x1): int(x2)]=blur
						#-- 인식된 객체 블러 처리 하는 코드
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        out.write(frame)
        cv2.imshow("Frame", resize_frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()