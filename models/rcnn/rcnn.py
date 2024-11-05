import torch, torchvision
import cv2, numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class rcnnDetector:
  def __init__(self, num_classes=80, conf_threshold=0.5):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # worry about num classes?
    if num_classes !=80:
      in_features = self.model.roi_heads.box_predictor.cls_score.in_features
      self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    self.model.eval()
    self.confidence_threshold = conf_threshold


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
      x1, y1, x2, y2 = map(int, box)
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(image, f'{label} {score:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image