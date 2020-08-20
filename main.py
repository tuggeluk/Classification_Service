from dwd_original.class_utils.classifier import dws_detector
import numpy as np
from flask import Flask, request, send_from_directory
from PIL import Image
import json
import sys

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

@app.route('/')
def hello_world():
    message = 'Welcome to the classifier'
    return message

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        print(request.headers)
        file = request.files['image']
        if file and allowed_file(file.filename):
            pic = Image.open(file).convert('L')
            pixels = np.asarray(pic)
            detect_list = detector.classify_img(pixels)
            detect_dict = dict(bounding_boxes = detect_list)
            print(json.dumps(detect_dict))
            return json.dumps(detect_dict)
        else:
            return 'Unsupported filetype'
    return
    '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SyntaxError("Insufficient arguments. Need to specify which classifier to use [dwd_original, dwd_v2, frcnn]")
    if sys.argv[1] == "dwd_original":
        print("Loading original dwd")
        detector = dws_detector()
    elif sys.argv[1] == "dwd_v2":
        print("Loading dwd version 2 (needs 20px interline!)")
    elif sys.argv[1] == "frcnn":
        print("Loading Faster Rcnn based classifier")
    else:
        raise SyntaxError("Available classifiers: [dwd_original, dwd_v2, frcnn], given argument: "+sys.argv[1])

    app.run(host='0.0.0.0')