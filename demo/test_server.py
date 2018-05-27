import requests
from PIL import Image
import numpy as np
import cv2
import json
from time import time
import sys
sys.path.append('../')
from class_utils.classifier import show_image
import os


command_url = "http://127.0.0.1:5000/classify"


dir = os.listdir(".")
dir = [x for x in dir if x[-4:] == ".png" ]

for img in dir:
    #resize to interline 10
    pic = Image.open(img).convert('L')
    pic = np.asanyarray(pic)
    #pic = cv2.resize(pic, None, None, fx=10.0/13.0, fy=10.0/13.0, interpolation=cv2.INTER_LINEAR)
    pic = Image.fromarray(pic)
    pic.save("tmp.png")

    with open(img, 'rb') as f:
        t = time()
        r = requests.post(command_url, files={"image": f})
        print(time()-t)
        print(r._content.decode("utf-8"))
        bbox_dict = json.loads(r._content.decode("utf-8"))

        show_image([np.asanyarray(Image.open("tmp.png"))], bbox_dict["bounding_boxes"], True, True)



