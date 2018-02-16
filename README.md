# Classification_Service
This repository contains a web service that finds and classifies symbols
of music notation. The service works `.png` and `.jpg` images that are scaled to a
staff interline of `10pt`.

## What is it made of ?
The service is based on the microframework `Flask`. The main route is `/classify`, it lets you upload an image (POST to image). 
The image will be scanned using a Deep Neural network and the bounding boxes of the detected symbols are returned in 
a `json` formatted string.

It is built to be used in conjunction with trained models following the `DeepWatershed-Detection` architecture. 
see: https://github.com/tuggeluk/DeepWatershedDetection 

## Contents
- `main.py` contains the flask app and calls the detection system located in `class_utils`.
- `class_utils/` contains the code that loads a pretrained tensorflow model and performs the detection.
- `demo/` contains a test/demo script for the service.
- **`important:`** for this service to work you need to create a folder called `trained_models` and 
place the correct pretrained models in said folder.  (can be found at: https://drive.google.com/open?id=1Knm26FjS6YMrBVU009mRTc19ims1oj6R)