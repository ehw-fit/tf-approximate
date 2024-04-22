import abc
import six

import tensorflow as tf

from python.keras.utils import quant_ops

@six.add_metaclass(abc.ABCMeta)
class Quantizer(object):
    @abc.abstractmethod
    def build(self, tensor_shape, name, layer):
        pass
    
    @abc.abstractmethod
    def __call__(self, inputs, training, weights, **kwargs):
        pass
    
    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError('Quantizer should implement get_config().')

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class _QuantizeHelper(object):
    def _add_range_weights(self, layer, name):
        min_weight = layer.add_weight(
            name + '_min',
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False)
        max_weight = layer.add_weight(
            name + '_max',
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False)
        
        return {'min_var': min_weight, 'max_var': max_weight}


class _FakeQuantizeHelper(object):
    def _create_range_vars(self, name):
        min_var = tf.Variable(
            name=name + '_min',
            initial_value=-6.0,
            trainable=False)
        max_var = tf.Variable(
            name=name + '_max',
            initial_value=6.0,
            trainable=False)
        
        return {'min_var': min_var, 'max_var': max_var}


class FakeQuantizer(_FakeQuantizeHelper, Quantizer):
    def __init__(self, num_bits, per_axis, symmetric, narrow_range):
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.symmetric = symmetric
        self.narrow_range = narrow_range
    
    def build(self, tensor_shape, name, layer):
        return self._create_range_vars(name)
    
    def __call__(self, inputs, training, weights, **kwargs):
        return quant_ops.FakeQuantize(
            inputs,
            weights['min_var'],
            weights['max_var'],
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
            symmetric=self.symmetric)
    
    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetrix': self.symmetric,
            'narrow_range': self.narrow_range
        }
    
    def __eq__(self, other):
        if not isinstance(other, FakeQuantizer):
            return False
        
        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric and
                self.narrow_range == other.narrow_range)
    
    def __ne__(self, other):
        return not self.__eq__(other)


class LastValueQuantizer(_QuantizeHelper, Quantizer):
    def __init__(self, num_bits, per_axis, symmetric, narrow_range):
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.symmetric = symmetric
        self.narrow_range = narrow_range
    
    def build(self, tensor_shape, name, layer):
        return self._add_range_weights(layer, name)
    
    def __call__(self, inputs, training, weights, **kwargs):
        return quant_ops.LastValueQuantize(
            inputs,
            weights['min_var'],
            weights['max_var'],
            is_training=training,
            num_bits=self.num_bits,
            per_channel=self.per_axis,
            symmetric=self.symmetric,
            narrow_range=self.narrow_range)
    
    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetrix': self.symmetric,
            'narrow_range': self.narrow_range
        }
    
    def __eq__(self, other):
        if not isinstance(other, LastValueQuantizer):
            return False
        
        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric and
                self.narrow_range == other.narrow_range)
    
    def __ne__(self, other):
        return not self.__eq__(other)


class MovingAverageQuantizer(_QuantizeHelper, Quantizer):
    def __init__(self, num_bits, per_axis, symmetric, narrow_range):
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.symmetric = symmetric
        self.narrow_range = narrow_range
    
    def build(self, tensor_shape, name, layer):
        return self._add_range_weights(layer, name)
    
    def __call__(self, inputs, training, weights, **kwargs):
        return quant_ops.MovingAvgQuantize(
            inputs,
            weights['min_var'],
            weights['max_var'],
            ema_decay=0.999,
            is_training=training,
            num_bits=self.num_bits,
            per_channel=self.per_axis,
            symmetric=self.symmetric,
            narrow_range=self.narrow_range)
    
    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetric': self.symmetric,
            'narrow_range': self.narrow_range
        }
    
    def __eq__(self, other):
        if not isinstance(other, MovingAverageQuantizer):
            return False
        
        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric and
                self.narrow_range == other.narrow_range)
    
    def __ne__(self, other):
        return not self.__eq__(other)


class AllValuesQuantizer(_QuantizeHelper, Quantizer):
    def __init__(self, num_bits, per_axis, symmetric, narrow_range):
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.symmetric = symmetric
        self.narrow_range = narrow_range
    
    def build(self, tensor_shape, name, layer):
        min_weight = layer.add_weight(
            name + '_min',
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)
        max_weight = layer.add_weight(
            name + '_max',
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)
        return {'min_var': min_weight, 'max_var': max_weight}
    
    def __call__(self, inputs, training, weights, **kwargs):
        return quant_ops.AllValuesQuantize(
            inputs,
            weights['min_var'],
            weights['max_var'],
            is_training=training,
            num_bits=self.num_bits,
            symmetric=self.symmetric,
            narrow_range=self.narrow_range)
    
    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetric': self.symmetric,
            'narrow_range': self.narrow_range
        }
    
    def __eq__(self, other):
        if not isinstance(other, AllValuesQuantizer):
            return False
        
        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric and
                self.narrow_range == other.narrow_range)
    
    def __ne__(self, other):
        return not self.__eq__(other)
