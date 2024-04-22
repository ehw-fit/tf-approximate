import numpy as np
import tensorflow as tf

from python.keras.utils.quantize_config import DefaultQuantizeConfig
from python.keras.utils import quantizers
from python.keras.layers.tunable_convolutional import (TunableApproxConv2DWithMinMaxVars, TunableApproxDepthwiseConv2DWithMinMaxVars)
from python.keras.layers.convolutional import (ApproxConv2DWithMinMaxVars, ApproxDepthwiseConv2DWithMinMaxVars)


class QuantCommonConfig(DefaultQuantizeConfig):
    def __init__(self, layer_type, kernel_name, num_bits, mul_table_file, fake_quant=False):
        super(QuantCommonConfig, self).__init__(num_bits, {0: (0, 1, 2)}, {kernel_name: (3, 4)})
        self.num_bits = num_bits
        self.mul_table_file = mul_table_file
        self.layer_type = layer_type
        self.fake_quant = fake_quant

        if fake_quant:
            self.input_quantizer = quantizers.FakeQuantizer(
                num_bits=num_bits, per_axis=False, symmetric=False, narrow_range=False)

            self.weight_quantizer = quantizers.FakeQuantizer(
                num_bits=num_bits, per_axis=False, symmetric=False, narrow_range=False)
    
    def get_substitute_layer(self, layer):
        config = layer.get_config()
        config['approx_num_bits'] = self.num_bits
        config['approx_mul_table_file'] = self.mul_table_file
        return self.layer_type.from_config(config)
    
    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'mul_table_file': self.mul_table_file,
            'fake_quant': self.fake_quant
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TunableQuantConv2DConfig(QuantCommonConfig):
    def __init__(self, num_bits, mul_table_file, fake_quant=False):
        super(TunableQuantConv2DConfig, self).__init__(
            TunableApproxConv2DWithMinMaxVars,
            'kernel',
            num_bits,
            mul_table_file,
            fake_quant)
    
    def set_substitute_layer_weights(self, layer, orig_layer):
        layer.set_weights(orig_layer.get_weights() + [np.bool_(False)])


class TunableQuantDepthwiseConv2DConfig(QuantCommonConfig):
    def __init__(self, num_bits, mul_table_file, fake_quant=False):
        super(TunableQuantDepthwiseConv2DConfig, self).__init__(
            TunableApproxDepthwiseConv2DWithMinMaxVars,
            'depthwise_kernel',
            num_bits,
            mul_table_file,
            fake_quant)
    
    def set_substitute_layer_weights(self, layer, orig_layer):
        layer.set_weights(orig_layer.get_weights() + [np.bool_(False)])


class QuantConv2DConfig(QuantCommonConfig):
    def __init__(self, num_bits, mul_table_file, fake_quant=False):
        super(QuantConv2DConfig, self).__init__(
            ApproxConv2DWithMinMaxVars,
            'kernel',
            num_bits,
            mul_table_file,
            fake_quant)
    
    def set_substitute_layer_weights(self, layer, orig_layer):
        layer.set_weights(orig_layer.get_weights())


class QuantDepthwiseConv2DConfig(QuantCommonConfig):
    def __init__(self, num_bits, mul_table_file, fake_quant=False):
        super(QuantDepthwiseConv2DConfig, self).__init__(
            ApproxDepthwiseConv2DWithMinMaxVars,
            'depthwise_kernel',
            num_bits,
            mul_table_file,
            fake_quant)
    
    def set_substitute_layer_weights(self, layer, orig_layer):
        layer.set_weights(orig_layer.get_weights())


tf.keras.utils.get_custom_objects()['QuantConv2DConfig'] = QuantConv2DConfig
tf.keras.utils.get_custom_objects()['QuantDepthwiseConv2DConfig'] = QuantDepthwiseConv2DConfig
tf.keras.utils.get_custom_objects()['TunableQuantConv2DConfig'] = TunableQuantConv2DConfig
tf.keras.utils.get_custom_objects()['TunableQuantDepthwiseConv2DConfig'] = TunableQuantDepthwiseConv2DConfig