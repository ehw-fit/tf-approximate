import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from tensorflow_model_optimization.python.core.quantization.keras import quantize, quantize_layout_transform, quantizers
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer, transforms
from tensorflow_model_optimization.python.core.quantization.keras.tflite.tflite_quantize_registry import TFLiteQuantizeProvider
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantizers
from keras.layers.convolutional import ApproxConv2D, ApproxConv2DWithMinMaxVars

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern


def _get_weights(bn_layer_node):
  """Returns weight values for fused layer, including copying original values in unfused version."""

  return collections.OrderedDict(
      list(bn_layer_node.input_layers[0].weights.items())
      + list(bn_layer_node.weights.items()))


def _get_layer_node(fused_layer, match_layer, weights):
  layer_config = tf.keras.layers.serialize(fused_layer)
  layer_config['name'] = layer_config['config']['name']
  # This config tracks which layers get quantized, and whether they have a
  # custom QuantizeProvider.
  layer_metadata = {'quantize_provider': match_layer.metadata['quantize_provider']}

  return LayerNode(layer_config, weights, metadata=layer_metadata)


class TestTransform(transforms.Transform):
    def pattern(self):
        return LayerPattern('Conv2D', {}, [])

    def replacement(self, match_layer):
        conv_layer = match_layer.layer

        approx_conv_layer = ApproxConv2DWithMinMaxVars(**dict(list(conv_layer['config'].items()) + []))

        return _get_layer_node(approx_conv_layer, match_layer, match_layer.weights)

    def custom_objects(self):
        return {'ApproxConv2DWithMinMaxVars': ApproxConv2DWithMinMaxVars}


class ApproxQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
    def apply(self, model, layer_quantize_map):
        # TODO: Sequential models not supported yet. Remove once support is added.
        # if isinstance(model, tf.keras.Sequential):
        #     return model, layer_quantize_map

        transforms = [
            TestTransform()
        ]

        return model_transformer.ModelTransformer(
            model, transforms,
            layer_quantize_map.keys(), layer_quantize_map).transform()


class ApproxConvWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""

  def __init__(self):
    """Construct LastValueQuantizer with params specific for TFLite Convs."""

    super(ApproxConvWeightsQuantizer, self).__init__(
        num_bits=8,
        per_axis=False,
        symmetric=True,
        narrow_range=True)

  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        shape=(),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)

    return [min_weight, max_weight]


class ApproxConvInputQuantizer(quantizers.MovingAverageQuantizer):
    def __init__(self):
        super(ApproxConvInputQuantizer, self).__init__(
            num_bits=8,
            per_axis=False,
            symmetric=True,
            narrow_range=True)

    def build(self, tensor_shape, name, layer):
        min_input = layer.add_weight(
            layer.name + '_input_min',
            shape=(),
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False)
        max_input = layer.add_weight(
            layer.name + '_input_max',
            shape=(),
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False)

        return [min_input, max_input]


class ConvQuantizeProvider(TFLiteQuantizeProvider):
    """QuantizeProvider for Conv2D/DepthwiseConv2D layers."""

    def __init__(self, weight_attrs, activation_attrs, quantize_output):
        super(ConvQuantizeProvider, self).__init__(weight_attrs, activation_attrs, quantize_output)

        self.weight_quantizer = ApproxConvWeightsQuantizer()
        self.input_quantizer = ApproxConvInputQuantizer()


tf.keras.utils.get_custom_objects()['ConvQuantizeProvider'] = ConvQuantizeProvider
tf.keras.utils.get_custom_objects()['ApproxConv2D'] = ApproxConv2D
tf.keras.utils.get_custom_objects()['ApproxConv2DWithMinMaxVars'] = ApproxConv2DWithMinMaxVars

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

seqmodel = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    quantize.quantize_annotate(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), quantize_provider=ConvQuantizeProvider(['kernel'], ['activation'], False)),
    tf.keras.layers.AveragePooling2D(),
    quantize.quantize_annotate(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'), quantize_provider=ConvQuantizeProvider(['kernel'], ['activation'], False)),
    tf.keras.layers.Flatten(),
    quantize.quantize_annotate(tf.keras.layers.Dense(128, activation='relu')),
    quantize.quantize_annotate(tf.keras.layers.Dense(84, activation='relu')),
    quantize.quantize_annotate(tf.keras.layers.Dense(10, activation='softmax', name='output'))

    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(10, activation='softmax')
])

# Convert to functional model
input_layer = tf.keras.layers.Input(batch_shape=seqmodel.layers[0].input_shape)
prev_layer  = input_layer
for layer in seqmodel.layers:
    layer._inbound_nodes = []
    prev_layer = layer(prev_layer)

model = tf.keras.models.Model([input_layer], [prev_layer])
model = quantize.quantize_apply(model, quantize_transform=ApproxQuantizeLayoutTransform())

# Compile the model
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Store graph as frozen
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='./tflogs',
                  name='mnist_graph.pbtxt',
                  as_text=True)

# Connect to Tensorboard and train the model
tensorboard = TensorBoard(log_dir="tflogs/{}".format(datetime.datetime.now().replace(microsecond=0).isoformat()))

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6, callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
