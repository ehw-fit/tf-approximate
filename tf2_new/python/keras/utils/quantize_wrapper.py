import tensorflow as tf
from tensorflow.python.util import tf_inspect
from python.keras.utils import utils
import numpy as np

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object

class QuantizeWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, quantize_config, **kwargs):

        if layer is None:
            raise ValueError('`layer` cannot be None.')

        self.orig_layer = layer
        layer = quantize_config.get_substitute_layer(layer)

        if 'name' not in kwargs:
            kwargs['name'] = self._make_layer_name(layer)

        super(QuantizeWrapper, self).__init__(layer, **kwargs)

        self.quantize_config = quantize_config

    def build(self, input_shape):
        super(QuantizeWrapper, self).build(input_shape)
        self.quantize_config.set_substitute_layer_weights(self.layer, self.orig_layer)

        self._weight_vars = []
        for weight, quantizer, pass_range in self.quantize_config.get_weights_and_quantizers(self.layer):
            quantizer_vars = quantizer.build(weight.shape, self._weight_name(weight.name), self)
            self._weight_vars.append((weight, quantizer, quantizer_vars, pass_range))
            self._trainable_weights.append(weight)
        
        self._input_vars = []
        for input_idx, quantizer, pass_range in self.quantize_config.get_input_quantizers(self.layer):
            quantizer_vars = quantizer.build(input_shape, 'input_{}'.format(input_idx), self)
            self._input_vars.append((input_idx, quantizer, quantizer_vars, pass_range))

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(self.layer.input_shape)
    
    def _make_quantizer_fn(self, quantizer, x, training, quantizer_vars):
        def quantizer_fn():
            return quantizer(x, training, weights=quantizer_vars)
        
        return quantizer_fn
    
    def call(self, inputs, training=None):

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # Quantize weights
        quantized_weights = []
        for weight, quantizer, quantizer_vars, pass_range in self._weight_vars:
            quantized_weight = utils.smart_cond(training,
                self._make_quantizer_fn(quantizer, weight, True, quantizer_vars),
                self._make_quantizer_fn(quantizer, weight, False, quantizer_vars))
            
            if pass_range:
                quantized_weights.append((quantized_weight, quantizer_vars['min_var'], quantizer_vars['max_var']))
            else:
                quantized_weights.append((quantized_weight))
        
        self.quantize_config.set_quantize_weights(self.layer, quantized_weights)
        
        # Quantize inputs
        quantized_inputs = []
        for input_idx, quantizer, quantizer_vars, pass_range in self._input_vars:
            quantized_input = utils.smart_cond(training,
                self._make_quantizer_fn(quantizer, inputs[input_idx], True, quantizer_vars),
                self._make_quantizer_fn(quantizer, inputs[input_idx], False, quantizer_vars))
            
            if pass_range:
                quantized_inputs.append((quantized_input, quantizer_vars['min_var'], quantizer_vars['max_var']))
            else:
                quantized_inputs.append((quantized_input))
        
        layer_inputs = self.quantize_config.quantize_inputs(self.layer, inputs, quantized_inputs, quantized_weights)

        args = tf_inspect.getfullargspec(self.layer.call).args
        if 'training' in args:
            outputs = self.layer.call(layer_inputs, training=training)
        else:
            outputs = self.layer.call(layer_inputs)
        
        return outputs
    
    def get_config(self):
        base_config = super(QuantizeWrapper, self).get_config()
        config = {'quantize_config': serialize_keras_object(self.quantize_config)}
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        config = config.copy()

        quantize_config = deserialize_keras_object(
            config.pop('quantize_config'),
            module_objects=globals(),
            custom_objects=None)

        layer = tf.keras.layers.deserialize(config.pop('layer'))

        return cls(layer=layer, quantize_config=quantize_config, **config)
    
    @property
    def trainable(self):
        return self.layer.trainable
    
    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value
    
    @property
    def trainable_weights(self):
        return self.layer.trainable_weights + self._trainable_weights
    
    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights + self._non_trainable_weights
    
    @property
    def updates(self):
        return self.layer.updates + self._updates
    
    @property
    def losses(self):
        return self.layer.losses + self._losses
    
    @staticmethod
    def _make_layer_name(layer):
        return '{}_{}'.format('approx', layer.name)
    
    @staticmethod
    def _weight_name(name):
        return name.split(':')[0].split('/')[-1]