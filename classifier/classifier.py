# classify img arrays
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pa



class dws_detector:
    model_path = "models"
    model_name = "RefineNet-Res101"
    mapping_path = "mappings.csv"
    seed = 123
    tf_session = None
    mapping = None

    patch_size = [640, 640]

    def __init__(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        print('Loading mappings ...')
        mapping = pa.read_csv(self.mapping_path)
        mapping = mapping[mapping["Init_name"] != "-1"]
        mapping = mapping.drop(mapping.columns[[0, 2]], axis=1)

        # strip values
        mapping['Init_name'] = mapping['Init_name'].str.strip()

        # reset index
        self.mapping = mapping.reset_index(drop=True)

        sess = tf.Session()
        print('Loading meta-graph')
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(self.model_path + "/" + self.model_name+".meta")
        print("Loading weights")
        saver.restore(sess, self.model_path + "/" + self.model_name)
        self.tf_session = sess



    def classify_img(self, img):
        # img dimesion (1,y,x,1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        predictions = model.predict_classes(img, batch_size=1)
        return predictions



if __name__ == '__main__':
    model = dws_detector()
    print("testing mode")
