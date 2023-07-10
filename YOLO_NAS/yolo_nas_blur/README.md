```python
while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            classname = int(cls)
            class_name = classNames[classname]
            if class_name in tartget_class:
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # classname = int(cls)
                # class_name = classNames[classname]
                conf = math.ceil((confidence*100))/100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1,x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] -3
                cv2.rectangle(frame, (x1, y1), c2, [255,144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                frame_area = frame[y1:y2, x1:x2]
                if frame_area.size != 0:  # 형상의 너비 또는 높이가 0인지 확인
                    blur = cv2.blur(frame_area, (20, 20))
                    frame[int(y1):int(y2), int(x1):int(x2)] = blur
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
```

### Step3. 결과확인

<img width="640" alt="스크린샷 2023-07-05 오후 11 10 08" src="https://github.com/junyong1111/ObjectDetection/assets/79856225/d29d8caf-c385-46b6-be1d-879c52531127">


일단 인식된 사람은 모두 블러 처리가 된걸 확인 할 수 있었다