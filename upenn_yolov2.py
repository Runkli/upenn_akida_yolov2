#! /usr/bin/env python
from sys import argv
import tensorflow as tf
from keras.models import load_model
from cnn2snn import check_model_compatibility
from keras.utils.vis_utils import plot_model
import cnn2snn 

if len(argv) != 2:
    print("pass model path: upenn_yolov2.py [h5 model path]")
    quit()

model_path = argv[1]

model_keras = load_model(model_path, compile = False)
model_keras.summary()

# compat = check_model_compatibility(model_keras, True)

# mode_keras = cnn2snn.transforms.sequentialize(model_keras)
# mode_keras = cnn2snn.transforms.fold_batchnorm(mode_keras)
# mode_keras = cnn2snn.transforms.syncretize(model_keras)

# model_keras.summary()
# compat = check_model_compatibility(model_keras, True)
plot_model(model_keras, to_file="keras_model.png", show_shapes=True)
