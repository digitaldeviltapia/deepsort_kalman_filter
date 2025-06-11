import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker


#video_path = os.path.join('D:/datasets/videos_vision/centro1_lq.MOV')
video_path = os.path.join('.', 'people.mp4')
video_out_path = os.path.join('D:/datasets/videos_vision/', 'people_yolo12n.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                           (frame.shape[1], frame.shape[0]))
#load a COCO-pretrained YOLO model (we use v8 n and m, and v12 n and m)
#model = YOLO("yolov8n.pt")
model = YOLO("yolo12n.pt")
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]
box_color = box_color = (255, 255, 0)  

detection_threshold = 0.5
#vamos a inicializar el conteo total de personas detectadas
#conteo_manual = 42 #centro1_lq
conteo_manual = 56 #people
#conteo_manual = 33 #centro2_lq
unique_ids = set()
while ret:

    results = model(frame, device='cpu')

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 0: # recordar que class_id 0 es persona
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            if hasattr(track, "track_id"):
                unique_ids.add(track.track_id)
                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 1)
            cv2.putText(frame, f'Track ID: {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cap_out.write(frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(20) 
    ret, frame = cap.read()

#para contar todo
CA = len(unique_ids)  #conteo automatico
CM = conteo_manual  #conteo manual
P = (1 - abs(CA - CM) / CM) * 100 if CM>0 else 0  #porcentaje de precision

print(f"\nConteo manual (CM): {CM}")
print(f"Conteo automático (CA): {CA}")
print(f"Precisión del conteo: {P:.2f}%")

cap.release()
cap_out.release()
cv2.destroyAllWindows()
