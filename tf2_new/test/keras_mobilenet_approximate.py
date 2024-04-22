import os
import sys
sys.path.append(os.path.join('..'))

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from python.keras.utils import quantize
from python.keras.utils.approx_quantize_config import (
    QuantConv2DConfig, QuantDepthwiseConv2DConfig,
    TunableQuantConv2DConfig, TunableQuantDepthwiseConv2DConfig)

from keras_applications.mobilenet import MobileNet
from keras_applications.mobilenet import (preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing import image


# cuDNN can sometimes fail to initialize when TF reserves all of the GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


backend_config = {
    'backend': tf.keras.backend,
    'layers':  tf.keras.layers,
    'models':  tf.keras.models,
    'utils':   tf.keras.utils
}

model = MobileNet(weights='imagenet', **backend_config)

layer_type_quantize_map = {
    tf.keras.layers.Conv2D: lambda: QuantConv2DConfig(8, '', True),
    tf.keras.layers.DepthwiseConv2D: lambda: QuantDepthwiseConv2DConfig(8, '', True)
}

model = quantize.quantize_annotate_model(model, layer_type_quantize_map)
model = quantize.quantize_apply(model)

img_path = '../examples/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x, **backend_config)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3, **backend_config)[0])