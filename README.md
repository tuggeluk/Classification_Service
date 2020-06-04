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
- `Dockerfile` The dockerized version of all of this to reduce the amount of maintanence and administration necessary

## Instructions on using the docker image
We assume you're in the home directory.

1. Build the docker image:

```bash
docker build dockerized_classifier/Detection_Service -t saty/detection
```

The `-t` argument tags the image with the label `saty/detection`

2. Create a self-restarting container from the image

```bash
docker create --restart unless-stopped --gpus all -p 5000:5000 --name detection_docker saty/detection
```

3. Start and enable the service associated with running the docker container

```bash
sudo service detection-docker start
sudo service detection-docker enable  # To auto-start the service on reboot
```

4. To turn of the service

```bash
sudo service detection-docker stop
sudo service detection-docker disable  # Only if you want to prevent the service from restarting on reboot
```