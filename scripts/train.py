import cv2, os, sys, argparse

from models.yolo.yolo import yoloDetector
from models.rcnn.rcnn import rcnnDetector
# from models.resnet.resnet import resnetDetector
# run as python -m scripts.train with flags

MODEL_MAP = {
  'yolo' : {
    'config': 'models/yolo/yolov4.cfg',
    'weights': 'models/yolo/yolov4.weights',
    'class_names': 'models/yolo/coco.names'
  },
  'yolo_hands' : {
    'config': 'models/yolo/hands/yolov4.cfg',
    'weights': 'models/yolo/hands/yolov4.weights',
    'class_names': 'models/yolo/hands/coco.names'
  },
  'resnet': {
    'config': 'models/resnet/resnet50.cfg',
    'weights': 'models/resnet/resnet50.weights',
    'class_names': 'models/resnet/coco.names'
  },
  'rcnn': {
    'num_classes': 80,
    'conf_threshold': 0.5
  }
}


def get_all_imgs(folder_path):
  images = []
  for root, _, files in os.walk(folder_path):
    for filename in files:
      if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(root, filename))
        images.append(img)
  
  return images


def eval_model(detector, images, output_folder):
  # create output dir if it dne
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  
  total_detects = 0
  for i, image in enumerate(images):
    detections = detector.detect(image)
    detector.draw_box(image, detections)
    total_detects += len(detections)
    cv2.imwrite(os.path.join(output_folder, f'{i}.jpg'), image)
    
  return total_detects


def parse_arguments():
  parser = argparse.ArgumentParser(description="Train a specific model.", add_help=False)
  subparsers = parser.add_subparsers(dest="model", required=True, help="Specify model to train")
  
  # parent parser (shared arguments)
  parent_parser = argparse.ArgumentParser(add_help=False)
  parent_parser.add_argument("-i", "--input", type=str, help="Path to input data", required=True)
  parent_parser.add_argument("-o", "--output", type=str, help="Path to output data", required=True)
    
  # yolo subparser
  yolo_parser = subparsers.add_parser("yolo", parents=[parent_parser], help="YOLO model requires config, weights, class_names")
  yolo_parser.add_argument("-c", "--config", type=str, default=MODEL_MAP['yolo']['config'],
                          help="Path to model config, ex. yolo/yolov4.cfg")
  yolo_parser.add_argument("-w", "--weights", type=str, default=MODEL_MAP['yolo']['weights'],
                          help="Path to model weights, ex. yolo/yolov4.weights")
  
  # resnet subparser
  
  # rcnn subparser
  rcnn_parser = subparsers.add_parser("rcnn", parents=[parent_parser], help="RCNN model requires num_classes, conf_threshold")
  rcnn_parser.add_argument("-t", "--conf_threshold", type=float, default=MODEL_MAP['rcnn']['conf_threshold'],
                          help="Confidence threshold, default is 0.5")
  
  return parser.parse_args()


def main():
  args = parse_arguments()
  
  if args.model == "yolo":
    detector = yoloDetector(args.config, args.weights, args.class_names)

  elif args.model == "resnet":
    detector = resnetDetector(args.config, args.weights, args.class_names)

  elif args.model == "rcnn":
    detector = rcnnDetector(args.conf_threshold)
  
  images = get_all_imgs(args.input)
  total_detects = eval_model(detector, images, args.output)
  print(f"Total detections: {total_detects}")

if __name__ == "__main__":
  # print(cv2.__version__)
  main()