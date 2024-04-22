import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import os
import sys
sys.path.append(os.path.join('..', 'extern/model-optimization'))
sys.path.append(os.path.join('..'))

from python.keras.layers.approx_convolutional import ApproxConv2DWithMinMaxVars
# ApproxConv2DWithMinMaxVars = tf.keras.layers.Conv2D

from tensorflow_model_optimization.python.core.quantization.keras import quantize

# Model Layout Transforms
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

# Model Layer Transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_transforms import _get_layer_node

class ApproxConv2DQuantize(transforms.Transform):
    def pattern(self):
        return transforms.LayerPattern('Conv2D', {}, [])
    
    def replacement(self, match_layer):
        conv_layer = match_layer.layer

        approx_conv_layer = ApproxConv2DWithMinMaxVars(**dict(list(conv_layer['config'].items()) + []))

        return _get_layer_node(approx_conv_layer, match_layer.weights)
    
    def custom_objects(self):
        return {'ApproxConv2DWithMinMaxVars': ApproxConv2DWithMinMaxVars}


class ApproxQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
    def apply(self, model, layer_quantize_map):
        transforms = [
            default_8bit_transforms.InputLayerQuantize(),
            default_8bit_transforms.SeparableConv1DQuantize(),
            default_8bit_transforms.SeparableConvQuantize(),
            default_8bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
            default_8bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
            default_8bit_transforms.Conv2DBatchNormReLUQuantize(),
            default_8bit_transforms.Conv2DBatchNormActivationQuantize(),
            default_8bit_transforms.Conv2DReshapeBatchNormQuantize(),
            default_8bit_transforms.Conv2DBatchNormQuantize(),
            default_8bit_transforms.ConcatTransform6Inputs(),
            default_8bit_transforms.ConcatTransform5Inputs(),
            default_8bit_transforms.ConcatTransform4Inputs(),
            default_8bit_transforms.ConcatTransform3Inputs(),
            default_8bit_transforms.ConcatTransform(),
            default_8bit_transforms.AddReLUQuantize(),
            default_8bit_transforms.AddActivationQuantize(),
            ApproxConv2DQuantize(),
        ]
        return model_transformer.ModelTransformer(
            model, transforms,
            set(layer_quantize_map.keys()), layer_quantize_map).transform()

# cuDNN can sometimes fail to initialize when TF reserves all of the GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Load and prepare the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# print(x_train.shape)

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32')  / 255

# print(x_train.shape)

y_train = y_train.astype('float32')
y_test  = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Define our model architecture
model = tf.keras.Sequential([
    quantize.quantize_annotate_layer(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:])),
    tf.keras.layers.AveragePooling2D(),
    quantize.quantize_annotate_layer(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model = quantize.quantize_apply(model, ApproxQuantizeLayoutTransform())

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Connect to Tensorboard and train the model
tensorboard = TensorBoard(log_dir="tflogs/{}".format(datetime.datetime.now().replace(microsecond=0).isoformat()))

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, callbacks=[tensorboard])

print('================================================================================')
print('Testing trained model...')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])