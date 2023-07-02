import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
from sort import *

cap = cv2.VideoCapture("video/people.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

count = 0
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCountUp = []
totalCountDown = []
limitup = [103, 161, 296, 161]
limitdown = [527, 489, 735, 489]

while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        detections = np.empty((0, 5))
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
            #label = f'{class_name}{conf}'
            #print("Frame N", count, "", x1, y1,x2, y2)
            #t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            #c2 = x1 + t_size[0], y1 - t_size[1] -3
            #cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            #cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            if class_name == "person" and conf > 0.3:
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))
        resultsTracker = tracker.update(detections)
        cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (255,0,0), 3)
        cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (255,0,0), 3)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 3)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            label = f'{int(id)}'
            t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] -3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            if limitup[0] < cx < limitup[2] and limitup[1] -15 < cy < limitup[3] + 15:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (0, 255, 0), 3)
            if limitdown[0] < cx < limitdown[2] and limitdown[1] -15 < cy < limitdown[3] + 15:
                if totalCountDown.count(id) == 0:
                    totalCountDown.append(id)
                    cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (0, 255, 0), 3)
        cv2.rectangle(frame, (100, 65), (441, 97), [255, 0, 255], -1, cv2.LINE_AA)

        cv2.putText(frame, str("Person Entering") + ":" + str(len(totalCountUp)), (141, 91), 0, 1, [255, 255, 255], thickness=2, lineType = cv2.LINE_AA)
        cv2.rectangle(frame, (710, 65), (1100, 97), [255, 0, 255], -1, cv2.LINE_AA)

        cv2.putText(frame, str("Person Leaving") + ":" + str(len(totalCountDown)), (741, 91), 0, 1, [255, 255, 255], thickness=2, lineType = cv2.LINE_AA)

        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()