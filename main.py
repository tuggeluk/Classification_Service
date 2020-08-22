import numpy as np
from flask import Flask, request, send_from_directory
from PIL import Image
import json
import sys

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
detector = None

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
        from dwd_original.class_utils.classifier import dws_detector
        detector = dws_detector()
    elif sys.argv[1] == "dwd_v2":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--scaling", type=int, default=1, help="scale factor applied to images after loading")
        dataset = 'DeepScoresV2'
        if dataset == 'MUSCIMA':
            parser.add_argument("--dataset", type=str, default='MUSCIMA',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
            parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_test",
                                help="dataset to perform inference on")
        elif dataset == 'DeepScoresV2':
            parser.add_argument("--dataset", type=str, default='DeepScoresV2',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
            parser.add_argument("--test_set", type=str, default="DeepScoresV2_2020_val",
                                help="dataset to perform inference on")
        elif dataset == 'DeepScores':
            parser.add_argument("--dataset", type=str, default='DeepScores',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
            parser.add_argument("--test_set", type=str, default="DeepScores_2017_test",
                                help="dataset to perform inference on")
        elif dataset == 'DeepScores_300dpi':
            parser.add_argument("--dataset", type=str, default='DeepScores_300dpi',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
            parser.add_argument("--test_set", type=str, default="DeepScores_300dpi_2017_val",
                                help="dataset to perform inference on, we use val for evaluation, test can be used only visually")
        elif dataset == 'Dota':
            parser.add_argument("--dataset", type=str, default='Dota',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
            parser.add_argument("--test_set", type=str, default="Dota_2018_debug",
                                help="dataset to perform inference on")
        elif dataset == 'VOC':
            parser.add_argument("--dataset", type=str, default='VOC',
                                help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota, VOC")
            parser.add_argument("--test_set", type=str, default="voc_2012_train",
                                help="dataset to perform inference on, voc_2012_val/voc_2012_train")
        parser.add_argument("--net_type", type=str, default="RefineNet-Res101",
                            help="type of resnet used (RefineNet-Res152/101)")
        parser.add_argument("--net_id", type=str, default="run_0",
                            help="the id of the net you want to perform inference on")

        parser.add_argument("--saved_net", type=str, default="backbone",
                            help="name (not type) of the net, typically set to backbone")
        parser.add_argument("--energy_loss", type=str, default="softmax", help="type of the energy loss")
        parser.add_argument("--class_loss", type=str, default="softmax", help="type of the class loss")
        parser.add_argument("--bbox_loss", type=str, default="reg",
                            help="type of the bounding boxes loss, must be reg aka regression")
        parser.add_argument("--debug", type=bool, default=False,
                            help="if set to True, it is in debug mode, and instead of running the images on the net, it only evaluates from a previous run")

        parser.add_argument("--individual_upsamp", type=str, default="True",
                            help="is the network built with individual upsamp heads")

        parsed = parser.parse_known_args()[0]

        path = "pretrained_models/RefineNet-Res101/run_0"
        from dwd_v2.main.dws_detector import DWSDetector
        detector = DWSDetector(imdb=None, path=path, pa=parsed, individual_upsamp=parsed.individual_upsamp)

        print("Loading dwd version 2 (needs 20px interline!)")
    elif sys.argv[1] == "frcnn":

        print("please use specific container")
        config_path = "fasterrcnn/configs/DeepScoresBaselines/faster_rcnn_v2/faster_rcnn_hrnetv2p_w32_1x_coco.py"
        pretrained_path = "fasterrcnn/work_dirs/faster_rcnn_hrnetv2p_w32_1x_coco/epoch_140.pth"
    else:
        raise SyntaxError("Available classifiers: [dwd_original, dwd_v2], given argument: "+sys.argv[1])

    app.run(host='0.0.0.0')