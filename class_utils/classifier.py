# classify img arrays
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pa
from models.dwd_net import build_dwd_net
from class_utils.dws_transform import perform_dws
import os.path as osp
from PIL import Image

class dws_detector:
    model_path = "trained_models_18_may"
    model_name = "RefineNet-Res101"
    mapping_name = "mappings.csv"

    interline = 10
    cached_bboxes = True

    seed = 123
    tf_session = None
    mapping = None
    overlap = 20
    root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

    patch_size = [640, 640]

    def __init__(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        print('Loading mappings ...')
        mapping = pa.read_csv(self.root_dir + "/" + self.mapping_name)
        mapping = mapping[mapping["Init_name"] != "-1"]
        mapping = mapping.drop(mapping.columns[[0, 2]], axis=1)

        # strip values
        mapping['Init_name'] = mapping['Init_name'].str.strip()

        # reset index
        self.mapping = mapping.reset_index(drop=True)

        # has to be adjusted according to the training scheme used
        self.energy_loss = "softmax"
        self.class_loss = "softmax"
        self.bbox_loss = "reg"


        sess = tf.Session()
        print('Loading model')
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 1],
                                    name='image_input')
        dws_heads, init_fn = build_dwd_net(self.input, model=self.model_name,
                                           num_classes=len(self.mapping),
                                           pretrained_dir="",
                                           substract_mean=False)
        [self.dws_energy, self.class_logits, self.bbox_size] = [dws_heads["stamp_energy"][self.energy_loss][-1], dws_heads["stamp_class"][self.class_loss][-1], dws_heads["stamp_bbox"][self.bbox_loss][-1]]

        # Name the tensors

        saver = tf.train.Saver(max_to_keep=1000)
        sess.run(tf.global_variables_initializer())
        print("Loading weights")
        saver.restore(sess, self.root_dir+"/" +self.model_path + "/" + self.model_name)
        self.tf_session = sess

    @staticmethod
    def add_name_to_tensor(tensor, name):
        return tf.identity(tensor, name=name)

    def classify_img(self, img):
        print("classify")
        if len(img.shape) < 4:
            img = np.expand_dims(np.expand_dims(img,-1),0)
        # offsets = dict(y = [-self.overlap], x = [-self.overlap])
        # while max(offsets["y"]) < img.shape[0]:
        #     offsets["y"].append(max(offsets["y"])+self.patch_size)
        # pad both axes to multiples of 320
        y_mulity = int(np.ceil(img.shape[1]/320.0))
        x_mulity = int(np.ceil(img.shape[2] / 320.0))
        canv = np.ones([y_mulity*320,x_mulity*320], dtype=np.uint8)*255
        canv = np.expand_dims(np.expand_dims(canv, -1), 0)

        canv[0,0:img.shape[1],0:img.shape[2]] = img[0]
        pred_energy, pred_class_logits, pred_bbox = self.tf_session.run(
            [self.dws_energy, self.class_logits, self.bbox_size],
            feed_dict={self.input: canv})
        pred_class = np.argmax(pred_class_logits, axis=3)

        dws_list = perform_dws(pred_energy, pred_class, pred_bbox, interline = self.interline, cached_bboxes=self.cached_bboxes, mapping=self.mapping)

        # dws_list, img = perform_dws(pred_energy, pred_class, pred_bbox, return_ccomp_img=True)
        # from PIL import Image
        # img.show()
        # binar_energy = (pred_energy <= 0) * 255
        # data_img = Image.fromarray(np.squeeze(binar_energy).astype(np.uint8))
        # data_img.show()

        for element in dws_list:
            element[4] = self.mapping.iloc[element[4]]["Symbol ID"]

        return dws_list


def show_image(data, gt_boxes=None, gt=False, text=False, save=False, name=""):
    from PIL import ImageDraw
    im = Image.fromarray(data[0].astype("uint8"))
    im.show()

    if gt:
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes:
            draw.rectangle(((row[0],row[1]),(row[2],row[3])), outline="red")
        #im.show()
    if text:
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes:
            draw.text((row[2],row[3]),row[4], fill="red")
        im.show()

    if save:
        im.save(name)

    return


if __name__ == '__main__':
    detection = dws_detector()
    from PIL import Image
    import cv2
    pic = Image.open("../demo/Bach_Fuge_C_DUR.png").convert('L')
    pic = np.asanyarray(pic)
    im = cv2.resize(pic, None, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_LINEAR)

    bboxes = detection.classify_img(im)
    show_image([np.asanyarray(im)], bboxes, True, True, True,
               "/root/test_image.png")
    print("testing mode")
