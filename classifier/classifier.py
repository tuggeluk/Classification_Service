# classify img arrays
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
import pdb

model_path = "models"
model_name = "model_28jul"
seed = 123
model = None

patch_size = [96, 48]

classes = [
    'Fingering_3',
    'TimeSignature_ 24',
    'Fingering_1',
    'Fingering_0',
    'Script_1fermata',
    'Art_Rest_[Duration 64 ]',
    'DynamicText_mf',
    'Art_Rest_[Duration 32 ]',
    'Fingering_2',
    'Art_Rest_[Duration 16 ]',
    'DynamicText_pp',
    'Script_-1staccato',
    'DynamicText_mp',
    'Flag_flags.u3',
    'Flag_flags.u6',
    'Flag_flags.u4',
    'Flag_flags.u5',
    'Art_Rest_[Duration 128 ]',
    'Script_1staccatissimo',
    'Script_1accent',
    'Flag_flags.d3',
    'Flag_flags.d5',
    'Fingering_5',
    'TimeSignature_ 54',
    'Flag_flags.d6',
    'Art_NoteHead_Full',
    'TimeSignature_ 22',
    'Art_NoteHead_Black',
    'Art_Rest_[Duration 2 ]',
    'DynamicText_ppp',
    'TimeSignature_ 68',
    'Script_-1fermata',
    'Art_Accidental_in_key_-1/2',
    'TimeSignature_ 44',
    'Fingering_4',
    'TimeSignature_ 64',
    'Art_Accidental_in_key_1/2',
    'TimeSignature_ 34',
    'Art_NoteHead_Long',
    'DynamicText_fff',
    'Art_Rest_[Duration 4 ]',
    'TimeSignature_ 58',
    'None_Class',
    'Clef_bass',
    'Art_Rest_[Duration 8 ]',
    'TimeSignature_ 38',
    'Accidental_-1',
    'Accidental_1/2',
    'TimeSignature_ 32',
    'Script_1marcato',
    'Script_1staccato',
    'DynamicText_sf',
    'Accidental_-1/2',
    'Script_1trill',
    'Art_NoteHead_Half',
    'Script_-1tenuto',
    'Flag_flags.d4',
    'Art_Accidental_in_key_0',
    'Script_-1accent',
    'DynamicText_f',
    'DynamicText_p',
    'Accidental_1',
    'Accidental_0',
    'Dots_1',
    'Script_-1marcato',
    'Clef_clefs.percussion at 0',
    'TimeSignature_ 78',
    'Script_1tenuto',
    'TimeSignature_ 128',
    'Clef_tenor',
    'Clef_treble',
    'DynamicText_ff',
    'TimeSignature_ 98',
    'Rest_[Duration 1 ]',
    'Script_-1staccatissimo'
]

mapping = {
    "Fingering_0":"CLUTTER",
    "Fingering_1":"CLUTTER",
    "Fingering_2":"CLUTTER",
    "Fingering_3":"CLUTTER",
    "Fingering_4":"CLUTTER",
    "Fingering_5":"CLUTTER",
    "TimeSignature_ 24":"TIME_TWO_FOUR",
    "TimeSignature_ 22":"TIME_TWO_TWO",
    "TimeSignature_ 54":"TIME_FIVE_FOUR",
    "TimeSignature_ 68":"TIME_SIX_EIGHT",
    "TimeSignature_ 44":"TIME_FOUR_FOUR",
    "TimeSignature_ 64":"CLUTTER",
    "TimeSignature_ 34":"TIME_THREE_FOUR",
    "TimeSignature_ 58":"CLUTTER",
    "TimeSignature_ 38":"TIME_THREE_EIGHT",
    "TimeSignature_ 32":"CLUTTER",
    "TimeSignature_ 78":"CLUTTER",
    "TimeSignature_ 128":"CLUTTER",
    "TimeSignature_ 98":"CLUTTER",
    "Flag_flags.u3": "FLAG_1_UP",
    "Flag_flags.d3": "FLAG_1",
    "Flag_flags.u4": "FLAG_2_UP",
    "Flag_flags.d4": "FLAG_2",
    "Flag_flags.u5": "FLAG_3_UP",
    "Flag_flags.d5": "FLAG_3",
    "Flag_flags.u6": "FLAG_4_UP",
    "Flag_flags.d6": "FLAG_4",
    "Clef_bass": "F_CLEF",
    "Clef_clefs.percussion at 0": "PERCUSSION_CLEF",
    "Clef_tenor": "C_CLEF",
    "Clef_treble": "Clef_treble",
    "Art_Rest_[Duration 64 ]":"ONE_64TH_REST",
    "Art_Rest_[Duration 32 ]":"ONE_32ND_REST",
    "Art_Rest_[Duration 16 ]":"ONE_16TH_REST",
    "Art_Rest_[Duration 128 ]":"ONE_128TH_REST",
    "Art_Rest_[Duration 2 ]":"HW_REST_set",
    "Art_Rest_[Duration 4 ]":"QUARTER_REST",
    "Art_Rest_[Duration 8 ]":"EIGHTH_REST",
    "Art_NoteHead_Full": "WHOLE_NOTE",
    "Art_NoteHead_Half":"NOTEHEAD_VOID",
    "Art_NoteHead_Long": "BREVE",
    "Art_NoteHead_Black": "NOTEHEAD_BLACK",
    "Art_Accidental_in_key_-1/2": "FLAT",
    "Art_Accidental_in_key_-1": "DOUBLE_FLAT",
    "Art_Accidental_in_key_0": "NATURAL",
    "Art_Accidental_in_key_1/2": "SHARP",
    "Art_Accidental_in_key_1": "DOUBLE_SHARP",
    "Accidental_-1/2":"FLAT",
    "Accidental_-1":"DOUBLE_FLAT",
    "Accidental_0":"NATURAL",
    "Accidental_1":"DOUBLE_SHARP",
    "Accidental_1/2":"SHARP",
    "DynamicText_p": "DYNAMICS_P",
    "DynamicText_pp": "DYNAMICS_PP",
    "DynamicText_mp": "DYNAMICS_MP",
    "DynamicText_f": "DYNAMICS_F",
    "DynamicText_ff": "DYNAMICS_FF",
    "DynamicText_sf": "DYNAMICS_SF",
    "DynamicText_mf": "DYNAMICS_MF",
    "DynamicText_ppp": "CLUTTER",
    "DynamicText_fff": "CLUTTER",
    "Script_1fermata":"DOT_set",
    "Script_-1fermata":"DOT_set",
    "Script_-1staccato":"DOT_set",
    "Script_1staccatissimo":"STACCATISSIMO",
    "Script_-1staccatissimo":"STACCATISSIMO",
    "Script_1accent":"ACCENT",
    "Script_-1accent":"ACCENT",
    "Script_1trill":"TR",
    "Script_-1marcato":"STRONG_ACCENT",
    "Script_1tenuto":"TENUTO",
    "Script_-1tenuto":"TENUTO",
    "Dots_1": "AUGMENTATION_DOT",
    "None_Class": "CLUTTER"
}

def init(model):

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
    return map_to_string(prediction)

def map_to_string(pred):
    return mapping[classes[int(pred)]]

def get_patch_size():
    return patch_size

model = init(model)
