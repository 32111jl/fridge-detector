# yolo for object detection (fridge items)

import cv2, numpy as np

class yoloDectector:
  def __init__(self, config_path, weights, class_names, conf_threshold=0.5, nms_threshold=0.4):
    self.net = cv2.dnn.readNetFromDarknet(config_path, weights)
    self.confidence_threshold = conf_threshold
    self.nms_threshold = nms_threshold
    
    with open(class_names, 'r') as f:
      self.classes = [line.strip() for line in f.readlines()]
    
    layer_names = self.net.getLayerNames()
    self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]


  def detect(self, image):
    blob = cv2.dnn.blobFromImage(image, (1 / 255.), (416, 416), (0, 0, 0), True, crop=False)
    self.net.setInput(blob)
    
    detections = self.net.forward(self.output_layers)
    return self.process_detections(image, detections)


  def process_detections(self, image, detections):
    h, w, _ = image.shape
    boxes = []
    confidences = []
    class_ids = []
    
    for output in detections:
      for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > self.confidence_threshold:
          center_x = int(detection[0] * w)
          center_y = int(detection[1] * h)
          
          width = int(detection[2] * w)
          height = int(detection[3] * h)
          
          x = int(center_x - width / 2)
          y = int(center_y - height / 2)
          
          boxes.append([x, y, width, height])
          confidences.append(float(confidence))
          class_ids.append(class_id)
    
    # non-maximal suppression so no repeat boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
    return [(self.classes[class_ids[i]], boxes[i]) for i in indices]


  def draw_box(self, image, detections):
    for detection in detections:
      label, box = detection
      x, y, w, h = box
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image