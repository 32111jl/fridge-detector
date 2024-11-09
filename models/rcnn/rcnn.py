# rcnn for object detecion (fridge items)

import torch, torchvision
import cv2, numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class rcnnDetector:
  def __init__(self, conf_threshold=0.5):
    self.class_names = self.load_class_names("models/rcnn/coco.names")
    num_classes = len(self.class_names)
    
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # put model in eval mode
    self.model.eval()
    self.confidence_threshold = conf_threshold


  def load_class_names(self, file_name):
    with open(file_name, 'r') as f:
      class_names = f.read().splitlines()
    
    return class_names


  def detect(self, image):
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
      outputs = self.model(image_tensor)[0]
    
    return self.process_detections(outputs, image)


  def process_detections(self, outputs, image):
    h, w, _ = image.shape
    boxes = []
    
    for i in range(len(outputs['scores'])):
      score = outputs['scores'][i].numpy()
      
      if score >= self.confidence_threshold:
        box = outputs['boxes'][i].numpy()
        label = outputs['labels'][i].item()
        score = score.item()
        boxes.append((label, box, score))
    
    return boxes


  def draw_box(self, image, detections):
    for detection in detections:
      label, box, score = detection
      class_name = self.class_names[label - 1]
      x1, y1, x2, y2 = map(int, box)
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(image, f'{class_name} {score:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image