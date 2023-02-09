import os
import glob
import time
import numpy as np
import cv2 as opencv
import tensorflow as tf

import coremltools
from coremltools.models.neural_network import NeuralNetworkBuilder, AdamParams

from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as KerasBackEnd

import uuid
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', action='store_true', help='provide the name of this face')

args = parser.parse_args()

print(tf.__version__)
# devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices([], 'GPU')
# print(tf.config.list_logical_devices('GPU'))

if args.name:
    print(f'Training mode: name = {args.name}')

def tripletLoss(label_true, label_pred, alpha=0.3):
    anchor, positive, negative = label_true[0], label_pred[1], label_pred[2]

    positiveDistance = tf.reduce_sum(
        tf.square(tf.subtract(anchor, positive)), axis=-1)
    negativeDistance = tf.reduce_sum(
        tf.square(tf.subtract(anchor, negative)), axis=-1)

    normalLoss = tf.add(tf.subtract(positiveDistance, negativeDistance), alpha)
    loss = tf.reduce_sum(tf.maximum(normalLoss, 0.0))

    return loss


def prepareFaceDatabase():
    database = {}

    for file in glob.glob("pos/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, model)

    return database


def faceClassification(image, database, model):
    encoding = img_to_encoding(image, model)
    # print(encoding)
    # print(encoding.shape)

    identity = None
    minDist = 100

    for (name, databaseEncoding) in database.items():
        dist = np.linalg.norm(databaseEncoding - encoding)
        if dist < minDist:
            minDist = dist
            identity = name
            
    print(f'minDist: {minDist}')

    if minDist > 0.50:
        return '0'
    else:
        return identity


def lockscreen():
    os.system('open -a ScreenSaverEngine')


def unlockscreen():
    os.system(command1)
    time.sleep(0.1)
    os.system(command2)
    time.sleep(0.1)
    os.system(command3)


def doFaceClassification():
    identity = None
    faceCount = 0
    ret, frame = frameCapture.read()
    grayImage = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)

    new_image = np.zeros(grayImage.shape, grayImage.dtype)
    new_image = cv2.convertScaleAbs(grayImage, alpha=1.0, beta=3.0)

    faces = faceCascade.detectMultiScale(new_image, 2, 2)

    for _ in faces:
        faceCount += 1

    if faceCount == 0:
        return None

    if faceCount == 1:
        for (x, y, w, h) in faces:
            global lastx
            global lasty
            global lastw
            global lasth
            global lastIdentity
            global repetition

            sameFace = (abs(x - lastx) + abs(y - lasty) /
                        2) < 75 and (abs(w - lastw) + abs(h - lasth)/2) < 75

            if sameFace and repetition >= 20:
                print("Skipped: " + lastIdentity)
                time.sleep(0.5)
                if 'phoom' in str(lastIdentity):
                    return '1'
                else:
                    return '0'
            elif sameFace is False:
                repetition = 0

            frame = opencv.rectangle(
                frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = frame[y: y + h, x: x + w]
            
            if args.name:
                cv2.imwrite(f'pos/{args.name}{uuid.uuid4()}.jpg', img)
                
            identity = faceClassification(img, faceDatabase, model)
            if 'phoom' in identity:
                identity = 'phoom'
            else:
                identity = '0'

            if lastIdentity == identity:
                repetition += 1
            else:
                repetition = 0
                lastIdentity = identity

            print(abs(x - lastx) + abs(y - lasty)/2)
            print("\n")
            print(abs(w - lastw) + abs(h - lasth)/2)
            print("\n")
            print(f'lastIdentity: {lastIdentity}')
            print('repetition: ' + str(repetition))

            lastx = x
            lasty = y
            lasth = h
            lastw = w

        if 'phoom' in str(identity):
            print(f'Yes')
            return '1'
        else:
            print('No')
            return '0'


KerasBackEnd.set_image_data_format('channels_first')

model = faceRecoModel(input_shape=(3, 96, 96))

print("setup successful")

model.compile(
    optimizer='adam',
    loss=tripletLoss,
    metrics=['accuracy']
)

load_weights_from_FaceNet(model)

def convert_keras_to_mlmodel(keras_url, mlmodel_url):
    # Only import things we need
    from keras.models import load_model
    keras_model = load_model(keras_url)

    keras_model.summary()

    class_labels = ['Valid', 'Invalid', 'Obstructed']
    mlmodel = coremltools.convert(
        keras_model,
        inputs=[coremltools.ImageType(channel_first=True, color_layout='BGR', scale=1.0/255)],
    )

    mlmodel.save(mlmodel_url)

def make_updatable(builder, model_url, mlmodel_updatable_path):
    model_spec = builder.spec
    builder.make_updatable(['sequential/dense/MatMul'])
    builder.set_category_cross_entropy_loss(name='lossLayer', input='sequential/dense/Softmax')
    builder.set_adam_optimizer(AdamParams(lr=1e-2, batch=32))
    builder.set_epoch(20)

    mlmodel_updatable = coremltools.models.MLModel(model_spec)
    mlmodel_updatable.save(mlmodel_updatable_path)

### SAVING ###

model.save('CustomFaceIDFaceNet')

config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"])

# Add another layer
h5_save_path = 'facenet.h5'
custom_model = tf.keras.Sequential()
custom_model.add(model)
custom_model.add(tf.keras.layers.Dense(3, activation='softmax'))

custom_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])

custom_model.save(h5_save_path)

# CoreML Model
coreml_model_path = 'IdentityClassifier.mlmodel'
convert_keras_to_mlmodel(h5_save_path, coreml_model_path)

spec = coremltools.utils.load_spec(coreml_model_path)
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
builder.inspect_layers(last=10)

builder.inspect_output_features()
neuralnetwork_spec = builder.spec

coreml_updatable_model_path = './UpdatableFacenet.mlmodel'
make_updatable(builder, coreml_model_path, coreml_updatable_model_path)

# TODO: use keras instead of tensorflow

exit(0)

print('Converted and saved to .mlmodel')
print('Reloading and adding layers')

model = coremltools.models.MLModel(save_name)
spec = model._spec

print([inp.name for inp in spec.description.input])
print([out.name for out in spec.description.output])
with open('model_spec.yaml', 'w+') as f:
    f.write(str(spec))

layer = spec.neuralNetwork.layers[-1]
print(layer)
labels = ['Valid', 'Invalid', 'Obstructed']

spec.neuralNetworkClassifier.stringClassLabels.vector.extend(labels)

# Builder
builder = NeuralNetworkBuilder(spec=model._spec)
builder.make_updatable(['out'])
builder.set_categorical_cross_entropy_loss(name='lossLayer', input='labelProbability')
builder.spec.description.trainingInput[0].shortDescription = 'Example Image'
builder.spec.description.trainingInput[1].shortDescription = 'True Label'

adam_params = AdamParams(lr=1e-3, batch=8, momentum=0)
adam_params.set_batch(8, [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
builder.set_adam_optimizer(adam_params)
builder.set_epochs(10, list(range(10, 1000, step=5)))

coremltools.utils.save_spec(builder.spec, 'CustomizableFacenet.mlpackage')


exit(0)

### SAVING ###

faceDatabase = prepareFaceDatabase()

faceCascade = opencv.CascadeClassifier('haarcascade_frontalface_default.xml')

lastx = 0
lasty = 0
lastw = 0
lasth = 0
repetition = 0
lastIdentity = '0'

command1 = """osascript -e 'tell application "system events" to keystroke return'"""
command2 = """osascript -e 'tell application "system events" to keystroke "Edifice@1213"'"""  # put the password here
command3 = """osascript -e 'tell application "system events" to keystroke return'"""

frameCapture = opencv.VideoCapture(0)

wrongFace = 0
noFace = 0

locked = False
ignoredStart = time.time()
currentIgnored = time.time()
lastFace = None

print("ready to go")

while True:
    face = doFaceClassification()

    if face == '1':
        ignoredStart = time.time()
        currentIgnored = ignoredStart
        if locked:
            unlockscreen()
            locked = False

    else:
        ignoredStart = time.time()

    timeIgnored = currentIgnored - ignoredStart
    timeIgnored = -timeIgnored

    if timeIgnored >= 10:  # 10 seconds delay before lock
        lockscreen()
        locked = True
