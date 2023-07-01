import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models

#-- GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
  print(torch.cuda.get_device_name(0))


#-- 사전학습된 Yolo_nas_small 모델 불러오기(빠르지만 정확도가 낮음)
model = models.get(Models.YOLO_NAS_S, pretrained_weights ="coco").to(device)
#-- 실시간 웹캠 코드
model.predict_webcam(conf =0.7)

#-- 모델을 ONNX foramt으로 변환
models.convert_to_onnx(model = model, input_shape = (3,640,640), out_path = "yolo_nas_s.onnx")


