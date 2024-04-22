import abc
import six

from python.keras.utils import quantizers


@six.add_metaclass(abc.ABCMeta)
class QuantizeConfig(object):
    @abc.abstractmethod
    def get_input_quantizers(self, layer):
        raise NotImplementedError('Must be implemented in subclasses')

    @abc.abstractmethod
    def quantize_inputs(self, layer, inputs, quantize_inputs, quantize_weights):
        raise NotImplementedError('Must be implemented in subclasses')

    @abc.abstractmethod
    def get_weights_and_quantizers(self, layer):
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def set_quantize_weights(self, layer, quantize_weights):
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError('QuantizeConfig should implement get_config().')

    @abc.abstractmethod
    def get_substitute_layer(self, layer):
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def set_substitute_layer_weights(self, layer, orig_layer):
        raise NotImplementedError('Must be implemented in subclasses.')



class DefaultQuantizeConfig(QuantizeConfig):
    def __init__(self, num_bits, inputs, weights):
        self.inputs = inputs   # {IN_ID: (QIN_ID, QIN_RANGE_MIN_ID, QIN_RANGE_MAX_ID), ...}
        self.weights = weights # {W_NAME: (QW_RANGE_MIN_ID, QW_RANGE_MAX_ID), ...}
        self.range_inputs_count = 2 * (sum(x is not None for x in inputs.values()) + \
                                       sum(x is not None for x in weights.values()))

        self.input_quantizer = quantizers.MovingAverageQuantizer(
            num_bits=num_bits, per_axis=False, symmetric=False, narrow_range=False)

        self.weight_quantizer = quantizers.LastValueQuantizer(
            num_bits=num_bits, per_axis=False, symmetric=False, narrow_range=False)
    
    def get_input_quantizers(self, layer):
        return [(input_idx, self.input_quantizer, len(input_spec) > 1) for input_idx, input_spec in self.inputs.items()]
    
    def quantize_inputs(self, layer, inputs, quantize_inputs, quantize_weights):
        layer_inputs = inputs + [None for _ in range(self.range_inputs_count)]

        for input_spec, quant_input in zip(self.inputs.items(), quantize_inputs):
            input_quant_desc = input_spec[1]
            layer_inputs[input_quant_desc[0]] = quant_input[0]

            if len(input_quant_desc) > 1:
                layer_inputs[input_quant_desc[1]] = quant_input[1]
                layer_inputs[input_quant_desc[2]] = quant_input[2]
        
        for weight_spec, quant_weight in zip(self.weights.items(), quantize_weights):
            weight_quant_desc = weight_spec[1]

            if len(weight_quant_desc) > 0:
                layer_inputs[weight_quant_desc[0]] = quant_weight[1]
                layer_inputs[weight_quant_desc[1]] = quant_weight[2]
        
        return layer_inputs
    
    def get_weights_and_quantizers(self, layer):
        return [(getattr(layer, weight_attr), self.weight_quantizer, len(weight_spec) > 0) for weight_attr, weight_spec in self.weights.items()]
    
    def set_quantize_weights(self, layer, quantize_weights):
        for weight_spec, quant_weight in zip(self.weights.items(), quantize_weights):
            setattr(layer, weight_spec[0], quant_weight[0])
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_config(self):
        return {
            'inputs': self.inputs,
            'weights': self.weights
        }
    
    def get_substitute_layer(self, layer):
        return layer
    
    def set_substitute_layer_weights(self, layer, orig_layer):
        pass
    
    def __eq__(self, other):
        if not isinstance(other, DefaultQuantizeConfig):
            return False
        
        return (self.inputs == other.inputs and
                self.input_quantizer == other.input_quantizer and
                self.weights == other.weights and
                self.weight_quantizer == other.weight_quantizer)
    
    def __ne__(self, other):
        return not self.__eq__(other)
