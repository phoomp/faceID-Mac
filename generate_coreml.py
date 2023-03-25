import os
from os.path import exists
from pathlib import Path

import argparse

import coremltools
from coremltools.models.neural_network import NeuralNetworkBuilder, AdamParams

import keras
from keras.models import load_model
from keras import backend as K

from fr_utils import *
from inception_blocks_v2 import *


# Set image data format
K.set_image_data_format('channels_first')

parser = argparse.ArgumentParser('Creates a FaceNet CoreML model')
parser.add_argument('-o', '--output', type=str, help='save name of the model', required=True)


def convert_keras_to_mlmodel(keras_model, mlmodel_url):
    print(keras_model.summary())

    class_labels = ['Valid', 'Invalid', 'Obstructed']
    mlmodel = coremltools.convert(
        keras_model,
        inputs=[coremltools.ImageType(channel_first=True, color_layout='BGR', scale=1.0/255)]
    )

    mlmodel.save(mlmodel_url)

def make_updatable(builder, model_url, mlmodel_updatable_path):
    model_spec = builder.spec
    builder.make_updatable(['ip_layer'])
    builder.set_category_cross_entropy_loss(name='lossLayer', input='sequential/dense/Softmax')
    builder.set_adam_optimizer(AdamParams(lr=1e-2, batch=32))

    mlmodel_updatable = coremltools.models.MLModel(model_spec)
    mlmodel_updatable.save(mlmodel_updatable_path)

def load_as_keras(path):
    return load_model(path)

def make_custom_model(original_model):
    custom_model = keras.Sequential()

    custom_model.add(original_model)
    # for layer in original_model.layers:
    #     custom_model.add(layer)

    # Classifier Output
    # custom_model.add(keras.layers.Dense(3, activation='softmax'))
    # custom_model.compile(loss=keras.losses.categorical_crossentropy,
    #                      optimizer=keras.optimizers.Adam(),
    #                      metrics=['accuracy']
    #                      )

    return custom_model


def main(args):
    # Display args
    mlmodel_save_path = args.output
    print(f'MLModel save path: {mlmodel_save_path}')

    # Create the model from fr_utils and inception_blocks_v2
    keras_model = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(keras_model)
    keras_model.save('tmp.h5')

    # Load as keras
    model = load_as_keras('tmp.h5')
    custom_model = make_custom_model(model)

    # Convert
    convert_keras_to_mlmodel(custom_model, mlmodel_save_path)

    # Preview and edit the converted model
    spec = coremltools.utils.load_spec(mlmodel_save_path)

    print(type(spec.neuralNetwork.layers[-2]))

    # Build
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

    # Define weights and biases
    # weights = np.random.random((128 * 3), type=np.uint8)
    # biases = np.random.random((3,))

    # builder.add_inner_product('ip_layer', W=weights, b=biases, input_channels=128, output_channels=3, has_bias=False, input_name='')
    builder.inspect_layers(last=10)

    nn_spec = builder.spec
    with open('spec.yaml', 'w+') as f:
        f.write(str(nn_spec))

    coreml_updatable_model_path = './FaceNetCustomized.mlmodel'
    make_updatable(builder, mlmodel_save_path, coreml_updatable_model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
