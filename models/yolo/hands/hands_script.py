# using google colab to train this instead, results in 'train' folder

import os
from ultralytics import *
# https://docs.ultralytics.com/models/yolov5/#performance-metrics

hand_model = YOLO('models/yolo/hands/train/weights/best.pt')
results = hand_model.predict(source='data/hands/Testing', imgsz=416, conf=0.95)

save_dir = 'data/hands/results/yolo_hands'
# results.save(save_dir=save_dir)
for i, result in enumerate(results):
  result.save(f"{save_dir}/pred_{i}.jpg")