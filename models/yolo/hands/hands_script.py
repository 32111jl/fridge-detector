import os
from ultralytics import *
# https://docs.ultralytics.com/models/yolov5/#performance-metrics

model = YOLO('yolov5mu.pt')

yaml_path = os.path.join('models', 'yolo', 'hands', 'data.yaml')
project_path = os.path.join('data', 'hands', 'results')

print(yaml_path, project_path)

model.train(data=yaml_path,
            epochs=10,
            batch=16,
            imgsz=640,
            save_dir=project_path)