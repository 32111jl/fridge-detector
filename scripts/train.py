import cv2, os, sys

from models.yolo.yolo import yoloDectector

# run as python -m scripts.train


def get_all_imgs(folder_path):
  images = []
  for root, _, files in os.walk(folder_path):
    for filename in files:
      if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(root, filename))
        images.append(img)
  
  return images


def eval_model(detector, images, output_folder):
  total_detects = 0
  for i, image in enumerate(images):
    detections = detector.detect(image)
    detector.draw_box(image, detections)
    total_detects += len(detections)
    cv2.imwrite(os.path.join(output_folder, f'{i}.jpg'), image)
    
  return total_detects


if __name__ == "__main__":
  print(cv2.__version__)
  config_path = 'models/yolo/yolov4.cfg'
  weights = 'models/yolo/yolov4.weights'
  class_names = 'models/yolo/coco.names'
  input_path = 'data/images/archive/fruits-360_dataset_100x100/fruits-360/Training/Apple Crimson Snow 1'
  output_path = 'models/yolo/output/apple_crimson'
  
  detector = yoloDectector(config_path, weights, class_names, conf_threshold=0.3, nms_threshold=0.4)
  images = get_all_imgs(input_path)
  total_detects = eval_model(detector, images, output_path)
  
  print(f'Total detections: {total_detects}')