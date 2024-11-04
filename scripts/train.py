import cv2, os, sys, argparse

from models.yolo.yolo import yoloDetector
# run as python -m scripts.train with flags

MODEL_MAP = {
  'yolo' : {
    'config': 'models/yolo/yolov4.cfg',
    'weights': 'models/yolo/yolov4.weights',
    'class_names': 'models/yolo/coco.names'
  },
  'resnet': {
    'config': 'models/resnet/resnet50.cfg',
    'weights': 'models/resnet/resnet50.weights',
    'class_names': 'models/resnet/coco.names'
  },
  'rcnn': {
    'config': 'models/rcnn/rcnn.cfg',
    'weights': 'models/rcnn/rcnn.weights',
    'class_names': 'models/rcnn/coco.names'
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
  total_detects = 0
  for i, image in enumerate(images):
    detections = detector.detect(image)
    detector.draw_box(image, detections)
    total_detects += len(detections)
    cv2.imwrite(os.path.join(output_folder, f'{i}.jpg'), image)
    
  return total_detects


def parse_arguments():
  parser = argparse.ArgumentParser(description="Train a specific model.")
  parser.add_argument("--model", choices=["yolo", "resnet", "rcnn"], help="Specify model to train", required=True)
  parser.add_argument("-i", "--input", type=str, help="Path to input data", required=True)
  parser.add_argument("-o", "--output", type=str, help="Path to output data", required=True)
  parser.add_argument("-c", "--config", type=str, help="Path to model config")
  parser.add_argument("-w", "--weights", type=str, help="Path to model weights")
  
  return parser.parse_args()


def main():
  args = parse_arguments()
  
  # check if -c and -w are provided; if not, use default paths
  if not args.config:
    args.config = MODEL_MAP[args.model]['config']
  if not args.weights:
    args.weights = MODEL_MAP[args.model]['weights']
  args.class_names = MODEL_MAP[args.model]['class_names']
  
  if args.model == "yolo":
    # from models.yolo.yolo import yoloDetector
    detector = yoloDetector(args.config, args.weights, args.class_names)

  elif args.model == "resnet":
    from models.resnet.resnet import resnetDetector
    detector = resnetDetector(args.config, args.weights, args.class_names)
    
  elif args.model == "rcnn":
    from models.rcnn.rcnn import rcnnDetector
    detector = rcnnDetector(args.config, args.weights, args.class_names)
  
  images = get_all_imgs(args.input)
  total_detects = eval_model(detector, images, args.output)
  print(f"Total detections: {total_detects}")

if __name__ == "__main__":
  # print(cv2.__version__)
  main()
  # config_path = 'models/yolo/yolov4.cfg'
  # weights = 'models/yolo/yolov4.weights'
  # class_names = 'models/yolo/coco.names'
  # input_path = 'data/images/archive/fruits-360_dataset_100x100/fruits-360/Training/Apple Crimson Snow 1'
  # output_path = 'models/yolo/output/apple_crimson'
  
  # detector = yoloDectector(config_path, weights, class_names, conf_threshold=0.3, nms_threshold=0.4)
  # images = get_all_imgs(input_path)
  # total_detects = eval_model(detector, images, output_path)
  
  # print(f'Total detections: {total_detects}')