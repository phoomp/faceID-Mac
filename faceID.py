import os
import glob
import time
import numpy as np
import cv2 as opencv
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as KerasBackEnd


def tripletLoss(label_true, label_pred, alpha=0.3):
    anchor, positive, negative = label_pred[0], label_pred[1], label_pred[2]

    positiveDistance = tf.reduce_sum(
        tf.square(tf.subtract(anchor, positive)), axis=-1)
    negativeDistance = tf.reduce_sum(
        tf.square(tf.subtract(anchor, negative)), axis=-1)

    normalLoss = tf.add(tf.subtract(positiveDistance, negativeDistance), alpha)
    loss = tf.reduce_sum(tf.maximum(normalLoss, 0.0))

    return loss


def prepareFaceDatabase():
    database = {}

    for file in glob.glob("pos2/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, model)

    return database


def faceClassification(image, database, model):
    encoding = img_to_encoding(image, model)

    identity = None
    minDist = 100

    for (name, databaseEncoding) in database.items():
        dist = np.linalg.norm(databaseEncoding - encoding)
        if dist < minDist:
            minDist = dist
            identity = name

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
                # print("Skipped: " + lastIdentity)
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

            # print(abs(x - lastx) + abs(y - lasty)/2)
            # print("\n")
            # print(abs(w - lastw) + abs(h - lasth)/2)
            # print("\n")
            # print(lastIdentity)
            # print(str(repetition))

            lastx = x
            lasty = y
            lasth = h
            lastw = w

        if 'phoom' in str(identity):
            return '1'
        else:
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

model.save('CustomFaceIDFaceNet.h5')

faceDatabase = prepareFaceDatabase()

faceCascade = opencv.CascadeClassifier('haarcascade_frontalface_default.xml')

lastx = 0
lasty = 0
lastw = 0
lasth = 0
repetition = 0
lastIdentity = '0'

command1 = """osascript -e 'tell application "system events" to keystroke return'"""
command2 = """osascript -e 'tell application "system events" to keystroke "Edifice@0970415531"'"""  # put the password here
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
