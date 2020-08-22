import requests
from PIL import Image
import numpy as np
import json
from time import time
import sys
sys.path.append('../')
from dwd_original.class_utils.classifier import show_image
import os
import cv2


command_url = "http://0.0.0.0:5000/classify"


dir = os.listdir("demo")
dir = ["demo/"+x for x in dir if x[-4:] == ".png" ]

for img in dir:
    #resize to interline 10
    pic = Image.open(img).convert('L')
    pic = np.asanyarray(pic)
    #pic = cv2.resize(pic, None, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    pic = Image.fromarray(pic)
    pic.save("tmp.png")

    with open("tmp.png", 'rb') as f:
        t = time()
        r = requests.post(command_url, files={"image": f})
        print(time()-t)
        print(r._content.decode("utf-8"))
        bbox_dict = json.loads(r._content.decode("utf-8"))

        show_image([np.asanyarray(Image.open("tmp.png"))], bbox_dict["bounding_boxes"], True, True)



