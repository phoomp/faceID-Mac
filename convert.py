import tensorflow as tf
import coremltools

from inception_blocks_v2 import *
from fr_utils import *

model = tf.keras.models.load_model('/Users/phoom/Documents/faceID-Mac/CustomFaceIDFaceNet')

model = coremltools.convert(model, convert_to='mlprogram')
model.save('facenet.mlmodel')
