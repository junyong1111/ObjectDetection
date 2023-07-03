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

#-- GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#-- 모델 설정
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
conf_treshold = 0.70
            
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

#-- deep sort start
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
cap.release()
out.release()
cv2.destroyAllWindows()
    

                
            
            






