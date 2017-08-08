import os
import base64
import uuid
from classifier import classifier
import numpy as np
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_PATCHES_ON_DISK'] = False

@app.route('/')
def hello_world():
    return 'Welcome to the classifier'

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        print(request.headers)
        file = request.files['image_patch']
        if file and allowed_file(file.filename):
            if (app.config['SAVE_PATCHES_ON_DISK']):
                path = writeFile(file)
                pic = Image.open(path).convert('L')
            else:
                pic = Image.open(file).convert('L')
            (width, height) = pic.size
            pixels = list(pic.getdata())
            pixels = np.array(pixels)
            pixels = pixels.reshape(height, width)
            return classifier.classify_img(pixels)
        else:
            return 'Unsupported filetype'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    '''

def writeFile(file):
    extension = os.path.splitext(file.filename)[1]
    filename = str(uuid.uuid4()) + extension
    filename = secure_filename(filename)
    path = app.config['UPLOAD_FOLDER'] + filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
