# Made by Phoom Punpeng
# Rewritten 12 Jan 2023

import argparse
from cryptography.fernet import Fernet

import glob
import time
import logging

import cv2
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as kb


parser = argparse.ArgumentParser()

parser.add_argument('--generate-key', dest='password', help='Generate and save a key to pwd from the specified password.')
parser.add_argument('--no-store', action='store_true', dest='no_store', help='Do not cache models and databases')
parser.add_argument('--db-path', dest='db_path', help='Specify the absolute path to the "pos" folder of the database. Defaults to pwd')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()


def main():
    if args.verbose:
        logging.Logger().setLevel()

    if args.generate_key is not None:
        logging.log()


if __name__ == '__main__':
    main()



