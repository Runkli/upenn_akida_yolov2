#! /usr/bin/env python
from sys import argv
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
from keras import layers
from cnn2snn import check_model_compatibility
from keras.utils.vis_utils import plot_model
import cnn2snn 
from utils import replace_intermediate_layer_in_keras, insert_intermediate_layer_in_keras, init_conv_id

if len(argv) != 2:
    print("pass model path: upenn_yolov2.py [h5 model path]")
    quit()

model_path = argv[1]

model_keras = load_model(model_path, compile = False)
#model_keras.summary()

#compat = check_model_compatibility(model_keras, True)

subseq_model = cnn2snn.transforms.sequentialize(model_keras) # keras model composed of sequential sub-models

plot_model(model_keras, to_file="imgs/keras_model.png", show_shapes=True)
plot_model(subseq_model, to_file="imgs/subseq_keras_model.png", show_shapes=True)

# get leaky_relu layer ids
def get_lrelu_ids(block):
    idxes = []
    for idx, layer in enumerate(block.layers):
        if(isinstance(layer, layers.LeakyReLU)):
            idxes.append(idx)

    return idxes
            
def make_compatible(block):
    # fold batchnorm
    # replace leaky_relu with relu
    block = cnn2snn.transforms.fold_batchnorm(block)
    leaky_relu_indices = get_lrelu_ids(block)
    first_lrelu = leaky_relu_indices[0]
    block = replace_intermediate_layer_in_keras(block, first_lrelu, layers.ReLU())
    leaky_relu_indices = get_lrelu_ids(block) # replace_intermediate_layer adds an internal layer, so we have to run it again since the indices changed
    #leaky_relu_indices = [i+1 for i in leaky_relu_indices]

    for i in leaky_relu_indices:
        block = replace_intermediate_layer_in_keras(block, i, layers.ReLU())

    return block

# test making one of the sequential sub-models compatible
print("BRANCH")
block = subseq_model.layers[3]
block.summary()

block = make_compatible(block)

# insert an input layer because it's implicit i think
block = insert_intermediate_layer_in_keras(block, 0, layers.InputLayer((26, 26, 512,)))

# insert a conv2d 1x1 kernel that hopefully doesn't alter the input to maxpool
block = insert_intermediate_layer_in_keras(block, 1, layers.Conv2D( kernel_size = 1, filters = 512, padding = 'same', kernel_initializer = 'ones') ) 

# insert relu because akida complains if there is no activation function, hopefully this doesn't alter the result much
block = insert_intermediate_layer_in_keras(block, 3, layers.ReLU() ) 
block.summary()


comp = check_model_compatibility(block, False)
print("comp?", comp)
#block.summary()



