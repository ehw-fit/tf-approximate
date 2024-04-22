import tensorflow as tf
import sys
sys.path.append("../")
sys.path.append("/home/michalpinos/tfa-approx_latest/tf-approximate-gpu/")
from python.keras.layers.fake_convolutional import FakeApproxSeparableConv2D, FakeApproxConv2D, FakeApproxDepthwiseConv2D
import keras.backend as K
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
trainX = trainX /255.0
testX  =testX / 255.0

ax_mult_path = "/home/michalpinos/tfa-approx_latest/tf-approximate-gpu/examples/axmul_8x8/mul8u_1JFF.bin"
input_layer = tf.keras.Input(shape=(32,32,3))
stem = FakeApproxConv2D(filters=16,kernel_size=(3,3), approx_num_bits=8, approx_mul_table_file=ax_mult_path)(input_layer)
x = FakeApproxSeparableConv2D(48, 3, padding="same", use_bias=False, approx_num_bits=8, approx_mul_table_file=ax_mult_path)(stem)
x = FakeApproxSeparableConv2D(48, 3, padding="same", use_bias=False, approx_num_bits=8, approx_mul_table_file=ax_mult_path)(x)
x = FakeApproxSeparableConv2D(48, 3, padding="same", use_bias=False, approx_num_bits=8, approx_mul_table_file=ax_mult_path)(x)
x = K.print_tensor(x, "sep: ")
x = FakeApproxConv2D(64, 3, padding="same", activation="relu", approx_num_bits=8, approx_mul_table_file=ax_mult_path)(x)
x = FakeApproxDepthwiseConv2D( 3, padding="same", dilation_rate=2, activation="relu", approx_num_bits=8, approx_mul_table_file=ax_mult_path)(x)
feat = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(10, activation="softmax")(feat)

# model = tf.keras.Sequential([
#     tf.keras.layers.SeparableConv2D(filters=6, kernel_size=(3, 3), activation='relu', padding="same",),
#     tf.keras.layers.SeparableConv2D(filters=16, kernel_size=(3, 3), activation='relu', padding="same",),
#     tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same",),
#     tf.keras.layers.SeparableConv2D(filters=48, kernel_size=(3, 3), activation='relu', padding="same",),
#     tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same",),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

model = tf.keras.Model(inputs=input_layer, outputs=out)
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="acc")
model.build(input_shape=(None,32,32,3))
model.summary()
# print(model.layers[0].weights)
# print(sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights]))
history_dsconv = model.fit(x=trainX, y=trainY, batch_size=1, steps_per_epoch=1, epochs=1, validation_data=(testX, testY), validation_steps=1)