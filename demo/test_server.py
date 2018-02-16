import requests
from PIL import Image
import numpy as np
import cv2
import json
from class_utils.classifier import show_image


command_url = "http://127.0.0.1:5000/classify"

img = "lg-9997209-aug-beethoven--page-2.png"

# resize to interline 10
pic = Image.open(img).convert('L')
pic = np.asanyarray(pic)
pic = cv2.resize(pic, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
pic = Image.fromarray(pic)
pic.save("tmp.png")

with open("tmp.png", 'rb') as f:
    r = requests.post(command_url, files={"image": f})
    print r._content
    bbox_dict = json.loads(r._content)

    show_image([np.asanyarray(Image.open("tmp.png"))], bbox_dict["bounding_boxes"], True, True)



