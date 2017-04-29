# classify img arrays
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
import pdb


model_path = "models"
model_name = "tmp"
seed = 123
model = None

patch_size = [128, 64]

# RELEVANT_CLASS = ['FLAT', 
#                                     'SHARP', 
#                                     'NATURAL', 
#                                     'G_CLEF',
#                                     'F_CLEF', 
#                                     'AUGMENTATION_DOT', 
#                                     'Clef_tenor',
#                                     'NOTEHEAD_BLACK', 
#                                     'NOTEHEAD_VOID', 
#                                     'WHOLE_NOTE',
#                                     'Script_marcato',
#                                     'TENUTO',
#                                     'FERMATA',
#                                     'ACCENT',
#                                     'STACCATO',
#                                     'HALF_REST',
#                                     'EIGHTH_REST', 
#                                     'QUARTER_REST',
#                                     'ONE_128TH_REST', 
#                                     'WHOLE_REST', 
#                                     'ONE_32ND_REST', 
#                                     'ONE_16TH_REST',
#                                     'TIME_TWO_FOUR', 
#                                     'TIME_FOUR_FOUR', 
#                                     'TIME_THREE_FOUR', 
#                                     'None_Class']


classes_array = [ 'TENUTO',
                                    'TIME_TWO_FOUR',
                                    'ONE_32ND_REST',
                                    'ONE_16TH_REST',
                                    'ONE_128TH_REST',
                                    'WHOLE_REST',
                                    'WHOLE_NOTE',
                                    'STACCATO',
                                    'TIME_FOUR_FOUR',
                                    'FERMATA',
                                    'TIME_THREE_FOUR',
                                    'NOTEHEAD_VOID',
                                    'HALF_REST',
                                    'EIGHTH_REST',
                                    'SHARP',
                                    'ACCENT',
                                    'Script_marcato',
                                    'FLAT',
                                    'QUARTER_REST',
                                    'NATURAL',
                                    'AUGMENTATION_DOT',
                                    'NOTEHEAD_BLACK',
                                    'G_CLEF',
                                    'Clef_tenor',
                                    'G_CLEF',
                                    'None_Class']


def init(model):
    # Workaround for keras bug
    # f = h5py.File(os.path.join(model_path, model_name), 'r+')
    # del f['optimizer_weights']
    # f.close()

    np.random.seed(seed)
    tf.set_random_seed(seed)

    print('Loading the model ...')
    model = load_model(os.path.join(model_path, model_name))
    return model


def classify_img(img):
    # img dimesion (1,y,x,1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict_classes(img, batch_size=1)
    return  map_to_string(prediction)


def map_to_string(pred):
    return classes_array[int(pred)]


def get_patch_size():
    return patch_size


model = init(model)
