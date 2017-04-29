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

@app.route('/')
def hello_world():
    return 'Welcome to the classifier'

@app.route('/classify', methods=['GET', 'POST'])
def upload_patch():
    if request.method == 'POST':
        print(request.headers)
        file = request.files['image_patch']
        extension = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + extension
        if file and allowed_file(filename):
            filename = secure_filename(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('.classify', image=filename))
        else:
            return 'no file attached'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/classify/<image>')
def classify(image):
    pic = Image.open(app.config['UPLOAD_FOLDER'] + image).convert('L')
    (width, height) = pic.size
    pixels = list(pic.getdata())
    pixels = np.array(pixels)
    pixels = pixels.reshape(height, width)
    return classifier.classify_img(pixels)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
