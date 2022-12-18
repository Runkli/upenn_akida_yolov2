import re
from keras.models import Model
from tensorflow import constant, float32, convert_to_tensor
import numpy as np


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)



    new_model = Model(inputs=model.inputs, outputs=x)
    return new_model


def insert_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(inputs=model.inputs, outputs=x)
    return new_model



def init_conv_id(shape, dtype=float32):
    print("init conv id shape", shape)
    # initializes 3x3 convolution weights as 1 for the center and 0 for the rest 
    #shape = (shape[0], shape[1], shape[2], 1)
    a = np.zeros(shape)
    mid = a.shape[0] // 2
    a[mid, mid, :, :] = 1
    
    tens = convert_to_tensor(a, dtype=np.float32)

    return tens 





