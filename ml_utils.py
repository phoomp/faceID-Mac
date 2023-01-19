import glob
import os.path

import cv2
import tensorflow as tf
import numpy as np

from fr_utils import *
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_true[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    raw_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(raw_loss, 0.0))  # Margin always 0 when performing inferences.

    return loss


def prepare_db(model, db_path='pos/*'):
    db = {}
    for file in glob.glob(db_path):
        identity = os.path.splitext(os.path.basename(file))[0]
        db[identity] = img_path_to_encoding(file, model)

    return db


if __name__ == '__main__':
    raise RuntimeError('This script is not meant to be run directly.')