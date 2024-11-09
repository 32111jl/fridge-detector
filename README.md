# Fridge Detector

This project benchmarks several modern computer vision models (adapted from OpenCV) to detect and track items in a fridge.

## Installation
1. Clone the repo
2. Install dependencies - `pip install -r requirements.txt`


## Models and Weights
#### YOLOv5mu:
- pre-trained weights were obtained from [Ultralytics Docs](https://docs.ultralytics.com/models/yolov5/#performance-metrics).
- using for quick-and-dirty hand detection.
  
#### YOLOv4:
- pre-trained weights and model configuration were obtained from [AlexeyAB's darknet repo](https://github.com/AlexeyAB/darknet).
- using for _____.

#### Resnet:
- pre-trained

#### R-CNN:
- pre-trained weights and model configuration were obtained from `torchvision`.


## Usage
Start training by running `scripts/train.py` with the desired model argument and flags. Specify the model and other required parameters, and ensure that input and output paths are correct:
```
python -m scripts.train <model_name> -i <input_path> -o <output_path> <additional_flags>
```
- `-m` helps run the module as a script
- `<model_name>`: the model to use (either "yolo", "resnet", or "rcnn")
- `-i`: the input folder path (path to the folder containing either testing or training data)
- `-o`: the output folder path (path to desired folder where annotated images will be contained)
- `<additional_flags>`: these are optional and model-specific, see below

#### YOLO arguments
- [optional] `-c`: the configuration file of the model (default is pre-trained)
- [optional] `-w`: the weights file of the model (default is pre-trained)

#### R-CNN arguments
- [optional] `-n`: the number of classes (default is 80)
- [optional] `-t`: the confidence threshold (default is 0.5)


### Example commands:
```
python -m scripts.train yolo -i data/images/New\ VegNet/1.\ Bell\ Pepper/Ripe -o models/yolo/output/bell_pepper/bell2

python -m scripts.train --model yolo -i data/images/archive/fruits-360_dataset_100x100/fruits-360/Training/Apple Crimson Snow 1 -o models/yolo/output/apple_crimson -c models/yolo/yolov4.cfg -w models/yolo/yolov4.weights

python -m scripts.train rcnn -i data/images/New\ VegNet/1.\ Bell\ Pepper/Ripe -o models/rcnn/output/bell_pepper/
```
